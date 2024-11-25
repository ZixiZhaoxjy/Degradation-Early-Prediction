# Decoupling Microscopic Degradation Patterns for Ultra-early Battery Prototype Verification Using Physics-informed Machine Learning
Manufacturing challenges and uncertainties have hindered the transition from battery prototypes to commercial products, making efficient quality verification essential. This work presents a physics-informed machine learning approach to quantify and visualize degradation patterns using electrical signals. The method enables non-destructive, temperature-adaptable lifetime predictions 25 times faster than traditional capacity tests, with 95.1% accuracy across temperatures. By interpreting statistical insights from material-agnostic features elicited through a multistep charging profile, it decouples degradation patterns. This approach supports sustainable management of defective prototypes, demonstrating the potential to unlock a $19.76$ billion USD scrap material recycling market in China by 2060. It accelerates quality verification, offering a path toward faster, more sustainable battery research and development.

# 1. Setup
## 1.1 Enviroments
* Python (Jupyter notebook) 
## 1.2 Python requirements
* python=3.11.5
* numpy=1.26.4
* tensorflow=2.15.0
* keras=2.15.0
* matplotlib=3.9.0
* scipy=1.13.1
* scikit-learn=1.3.1
* pandas=2.2.2

# 2. Datasets
* Prototype ternary nickel manganese cobalt lithium-ion batteries were cycled under controlled temperatureconditions (25, 35, 45, 55℃) under multi-step charging (0.33C to 3C, where 1C is 1.1A) and 1C constant discharge beyond EOL thresholds. We generate a unique battery prototype verification dataset spanning lifetimes of 480 to 1025 cycles (average lifetime of 775 with a standard deviation of 175 under EOL80 definition).
* Raw and processed datasets have been deposited in TBSI-Sunwoda-Battery-Dataset, which can be accessed at [TBSI-Sunwoda-Battery-Dataset](https://github.com/terencetaothucb/TBSI-Sunwoda-Battery-Dataset). 


# 3. Experiment
## 3.1 Settings
* In the code of each model, there are options to change parameters at the very beginning. For example, in the ChemicalProcessModel.py, the following parameters can be modified to adjust the training process.
```python
learning_rates = [3e-4, 1e-4]
# We train the model using Adam as optimizer, and epochs 30, learning rate 1e-4, mse loss with L1 regularization
lr_losses = {}
best_lr = None
best_loss = float('inf')
best_model_state = None

train_epochs = 100
raw_data = pd.read_csv("./raw_data_0920.csv")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_dataset = BattDataset(raw_data, train=True)
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
valid_dataset = BattDataset(raw_data, train=True)
valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False)
test_dataset = BattDataset(raw_data, train=False)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
criterion = nn.MSELoss().to(device)

```
## 3.2 Run
* After changing the experiment settings, __run `***model.py` directly for training and testing.__  

# 4. Experiment Details
The entire experiment consists of three steps as well as three models: 
* ChemicalProcessModel
* DomainAdaptModel
* DegradationTrajectoryModel

First, we model multi-dimensional chemical processes using early cycle and guiding sample data; second, we adapt these predictions to specific temperatures; and third, we use adapted chemical processes to avoid the need for physical measures in later cycles. The extent of early data used is tailored to meet the desired accuracy, assessed by mean absolute percentage error for consistent cross-stage comparisons.

## 4.1 Chemical process prediction model considering initial manufacturing variability (ChemicalProcessModel)
The **ChemicalProcessModel** predicts chemical process variations by using input voltage matrix $U$. Given a feature matrix $\mathbf{F} \in \mathbb{R}^{(C \times m) \times N}$ (see paper for more details on the featurization taxonomy), where $N$ is the number of features, the model learns a composition of $L$ intermediate layers of a neural network:

$\hat{\mathbf{F}} = f_\theta(U) = \left(f_\sigma^{(L)} \left(f_\theta^{(L)} \circ \cdots \circ f_\sigma^{(1)} \left(f_\theta^{(1)}\right)\right)\right)(U)$

where $L = 3$ in this work. $\hat{\mathbf{F}}$ is the output feature matrix, i.e., $\hat{\mathbf{F}} \in \mathbb{R}^{(C \times m) \times N}$, $\theta = \{\theta^{(1)}, \theta^{(2)}, \theta^{(3)}\}$ is the collection of network parameters for each layer, $U \in \mathbb{R}^{(C \times m) \times 10}$ is the broadcasted input voltage matrix, and $f_\theta(U)$ is a neural network predictor. All layers are fully connected. The activation function used is Leaky ReLU (leaky rectified linear unit), denoted as $f_\sigma$. 

Here is the implementation:
```python
 class ChemicalProcessModel(nn.Module):
    def __init__(self):
        super(ChemicalProcessModel, self).__init__()
        self.fc1 = nn.Linear(10, 32)
        self.fc2 = nn.Linear(32, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 42)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x
```
In each selected temperature, we split the data into 75% and 25% for training and testing, respectively. We train the chemical process prediction model using the Adam optimizer, with 30 epochs and a learning rate of \(10^{-4}\). The loss function of the chemical process prediction model is defined as:

$L_{\text {loss_ChemicalProcess }}=\frac{\sum_{i=1}^{C}\left(\boldsymbol{F}_{i}-\hat{\boldsymbol{F}}_{i}\right)^{2}}{C}+\lambda_{1} * \sum_{i=1}^{C}\left|\boldsymbol{F}_{i}-\hat{\boldsymbol{F}}_{i}\right|$
where $F_i$ is the $i-th$ label of the defined chemical processes, $hat{{F}_i}$ is the predicted chemical process feature matrix for the \(i\)-th cycle, $lambda_1$ is the regularization parameter, set to \(10^{-5}\).


See the Methods section of the paper for more details.
## 4.2 Latent space scaling and sampling to generate the data
After training the VAE model, it is necessary to sample its latent space to generate new data. This section will specifically explain how to perform scaling and sampling in the latent space.
### 4.2.1 Latent space scaling informed by prior knowledge
Certain retirement conditions, e.g., extreme SOH and SOC can be under-represented in the battery recycling pretreatment due to practical constraints. Specifically, the retired batteries exhibit concentrated SOH and SOC, leading to poor estimation performance when confronted with out-of-distribution (OOD) batteries.  This phenomenon results from the fact that retired electric vehicle batteries are collected in batches with similar historical usages and, thus similar SOH conditions. With a stationary rest following, the voltage values of the collected retired batteries are discharged lower than a certain threshold due to the safety concerns of the battery recyclers, resulting in a stationary rest SOC lower than 50%. Even if the explicit battery retirement conditions are still unknown, we can use this approximated prior knowledge to generate enough synthetic data to cover the actual retirement conditions. 

Given two data generation settings, namely, interpolation and extrapolation, we use different latent space scaling strategies. In the interpolation setting, the scaling matrix $T$ is an identity matrix $I$ assuming the encoder network and decoder network can learn the inherited data structures without taking advantage of any prior knowledge. In the extrapolation setting, however, the assumption cannot be guaranteed due to the OOD issue, a general challenge of machine learning models. Here we use the means of training and testing SOC distributions to define the scaling matrix, a prior knowledge of the battery retirement conditions, then the latent space is scaled as:
$$z_{\text{mean}}^{'} = T_{\text{mean}} \cdot z_{\text{mean}}$$
$$z_{\text {log-var }}^{'}=T_{\text {log-var}} \cdot z_{\text {log-var}}$$
where, $T_{\text{mean}}$ and $T_{\text{log-var}}$ are the scaling matrices defined by the broadcasted mean, and variance ratio between the testing and training SOC distributions. We emphasize that the SOH distributions are irrelevant to such a scaling. This is because these identical SOH values could be seen as representing physically distinct batteries, i.e., they do not affect the scaling process. Thus, feeding the model with the same SOH values during training and reconstruction does not present an OOD problem. On the other hand, for the SOC dimension, our goal is to generate data under unseen SOC conditions, where physical tests cannot be exhausted.
### 4.2.2 Sampling in the scaled latent space 
The sampling step in the VAE is a bridge between the deterministic output of the encoder neural network and the stochastic nature of the scaled latent space. It allows the model to capture the hidden structure of the input data, specifically the pulse voltage response $x$ and $cond$ to explore similar data points. The sampling procedure can be formulated as:
$$z = z_{\text{mean}} + e^{\frac{1}{2}z_{\text {log-var }}} \cdot \boldsymbol{\epsilon}$$
where, $\boldsymbol{\epsilon}$, is a Gaussian noise vector sampled from $\boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$. The exponential term $e^{\frac{1}{2}z_{\text {log-var }}}$ turns the log variance vector to a positive variance vector. $z$ is the sampled latent variable. 

The implementation of data generation process based on latent space scaling and sampling is as follows.
```python
def generate_data(vae, train_features, train_condition, test_condition, encoder, decoder, sampling_multiplier, batch_size, epochs, latent_dim):
    # Normalize feature data (training)
    feature_scaler = MinMaxScaler().fit(train_features)
    train_features_normalized = feature_scaler.transform(train_features)

    # Combine training and testing conditional data for scaling
    combined_conditions = np.vstack([train_condition, test_condition])
    # Normalize conditional data (training and testing using the same scaler)
    condition_scaler = MinMaxScaler().fit(combined_conditions)
    train_condition_normalized = condition_scaler.transform(train_condition)
    test_condition_normalized = condition_scaler.transform(test_condition)
    # Fit the VAE model using training data
    history = vae.fit([train_features_normalized, train_condition_normalized], train_features_normalized,
                      epochs=epochs, batch_size=batch_size, verbose=0)
    # Generate new samples based on testing conditions
    num_samples = len(test_condition_normalized) * sampling_multiplier
    print("num_samples",num_samples)
    random_latent_values_new = K.random_normal(shape=(num_samples, latent_dim), seed=0)
    random_latent_values_train = K.random_normal(shape=(len(train_condition_normalized) * sampling_multiplier, latent_dim), seed=0)

    # Use the testing conditional input for generating data
    repeated_conditions = np.repeat(test_condition_normalized, sampling_multiplier, axis=0)

    new_features_normalized = decoder.predict([random_latent_values_new, repeated_conditions])

    # Denormalize the generated feature data
    generated_features = feature_scaler.inverse_transform(new_features_normalized)

    repeated_conditions_train = np.repeat(train_condition_normalized, sampling_multiplier, axis=0)

    train_features_normalized = decoder.predict([random_latent_values_train, repeated_conditions_train])

    # Denormalize the generated feature data
    train_generated_features = feature_scaler.inverse_transform(train_features_normalized)

    train_generated_features = np.vstack([train_generated_features, generated_features])

    # Denormalize the repeated conditions to return them to their original scale
    repeated_conditions_denormalized = condition_scaler.inverse_transform(repeated_conditions)
    # Combine generated features with their corresponding conditions for further analysis
    generated_data = np.hstack([generated_features, repeated_conditions_denormalized])

    return generated_data, generated_features, repeated_conditions_denormalized, history, train_generated_features
```
## 4.3 Random forest regressor for SOH estimation
Since the data has been generated, the next step is to use the generated data to predict the SOH. We use the generated data to train a random forest model to predict SOH，and the random forest for regression can be formulated as:
$$\overline{y} = \overline{h}(\mathbf{X}) = \frac{1}{K} \sum_{k=1}^{K} h(\mathbf{X}; \vartheta_k, \theta_k)$$
where $\overline{y}$ is the predicted SOH value vector. $K$ is the tree number in the random forest. $\vartheta_k$ and $\theta_k$ are the hyperparameters. i.e., the minimum leaf size and the maximum depth of the $k$ th tree in the random forest, respectively. In this study, the hyperparameters are set as equal across different battery retirement cases, i.e., $K=20$ , $\vartheta_k=1$, and $\theta_k=64$, for a fair comparison with the same model capability. 

The Implementations are based on the ensemble method of the Sklearn Package (version 1.3.1) in the Python 3.11.5 environment, with a random state at 0.
```python
    # Phase 2: Train Model on Generated Data for Selected Testing SOC
    model_phase2 = RandomForestRegressor(n_estimators=20,max_depth=64,bootstrap=False).fit(X_generated, SOH_generated)
    y_pred_phase2 = model_phase2.predict(X_test)
    mape_phase2, std_phase2 = mean_absolute_percentage_error(y_test, y_pred_phase2)
```
# 5. Access
Access the raw data and processed features [here](https://zenodo.org/uploads/11671216) under the [MIT licence](https://github.com/terencetaothucb/Pulse-Voltage-Response-Generation/blob/main/LICENSE). Correspondence to [Terence (Shengyu) Tao](terencetaotbsi@gmail.com) and CC Prof. [Xuan Zhang](xuanzhang@sz.tsinghua.edu.cn) and [Guangmin Zhou](guangminzhou@sz.tsinghua.edu.cn) when you use, or have any inquiries.
# 6. Acknowledgements
[Terence (Shengyu) Tao](mailto:terencetaotbsi@gmail.com) and [Zixi Zhao](zhaozx23@mails.tsinghua.edu.cn)  at Tsinghua Berkeley Shenzhen Institute designed the model and algorithms, developed and tested the experiments, uploaded the model and experimental code, revised the testing experiment plan, and wrote this instruction document based on supplementary materials.  

