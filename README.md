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
In each selected temperature, we split the data into 75% and 25% for training and testing, respectively. We train the chemical process prediction model using the Adam optimizer, with 30 epochs and a learning rate of \(10^{-4}\). The loss function of the chemical process prediction model is implementated as:
```python
criterion = nn.MSELoss().to(device)
def loss_fn(outputs, labels, model, l1_strength):
        loss = criterion(outputs, labels)
        l1_regularization = add_l1_regularization(model, l1_strength)
        loss += l1_regularization
        return loss
```
See the Methods section of the paper for more details.
## 4.2 Multi-domain adaptation
After training the VAE model, it is necessary to sample its latent space to generate new data. This section will specifically explain how to perform scaling and sampling in the latent space.
### 4.2.1 Physics-informed transferability metric
It is time-consuming and cost-intensive to enumerate continuous temperature verifications, we therefore formulate a knowledge transfer from existing measured data (source domain) to arbitrary intermediate temperatures (target domain). The transfer is compatible with multi- and uni-source domain adaptation cases for tailored verification purposes. Here we use a multi-source domain adaptation to elucidate the core idea. For instance, we take 25, 55℃ as source domains and 35, 45℃ as target domains. 

We propose a physics-informed transferability metric to quantitatively evaluate the effort in the knowledge transfer. The proposed transferability metric integrates prior physics knowledge inspired by the Arrhenius equation:

$$
r = A e^{-\frac{E_a}{k_B T}} 
$$

where: $A$ is a constant, $r$ is the aging rate of the battery, $E_a$ is the activation energy, $k_B$ is the Boltzmann constant, and $T$ is the Kelvin temperature.

The Arrhenius equation provides us with important information that the aging rate of batteries is directly related to the temperature. Therefore, the Arrhenius equation offers valuable insights into translating the aging rate between different temperatures. We observe the domain-invariant representation of the aging rate ratio, consequently, the proposed Arrhenius equation-based transferability metric $AT_{score}$ is defined as:

$$
AT_{score}  = \frac{r_{target}}{r_{target}} = \frac{e^{-\frac{E_a^s}{k_B T_s}}}{e^{-\frac{E_a^t}{k_B T_t}}} 
$$

where $E_a^s$ is the activation energy of the source domain, $E_a^t$ is the activation energy of the target domain, $T_s$ and $T_t$ are the Kelvin temperatures of the source domain and the target domain, respectively.

The closer the $AT_{score}$ is to 1, the more similar the source domain and target domain are, so the better the knowledge transfer is expected. Since the dominating aging mechanism is unknown (characterized by $E_a$) as a posterior, we alternatively determine the aging rate by calculating the first derivative concerning the variations on the predicted chemical process curve:

$$
r = \frac{d\hat{F}}{dC} \tag{7}
$$

where $\hat{F}$ is the predicted chemical process feature matrix. We linearize the calculation in adjacent cycles by sampling the point pairs on the predicted chemical process:

$$
r = \frac{\sum_{i=0}^n \left(F_{\text{end}+i} - F_{\text{start}+i}\right)}{n \cdot (\text{end} - \text{start})}
$$

where $n$ is the number of point pairs, $start$ and $end$ are the cycle indices where we start and end the sampling, respectively, $F_{end+i}$ and $F_{start+i}$ are the feature values for the $(start+i)th$ and $(end+i)th$ cycles, respectively.

This calculation mitigates the noise-induced errors, resulting in a more robust aging rate computation. For domains where the aging mechanism is already known (different domains share the same \( E_a \)), the $AT_{score}$ can be expressed in the following form:

$$
\text{AT}_{\text{score}}^{source \to target} = e^{\frac{E_a}{k_B} \left( \frac{1}{T_t} - \frac{1}{T_s} \right)} 
$$

where $\frac{E_a}{k_B}$ is a constant value. This formula ensures that, in cases where the aging mechanism is known, we can calculate transferability between different domains using only the temperatures of the source and target domains. This allows the model for continuous temperature generalization without any target data.

### 4.2.2 Multi-domain adaptation using the physics-informed transferability metric

### 4.2.3 Chain of degradation

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

