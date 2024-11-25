# Decoupling Microscopic Degradation Patterns for Ultra-early Battery Prototype Verification Using Physics-informed Machine Learning
Manufacturing complexities and uncertainties have impeded the transition from material prototypes to commercial batteries, making prototype verification critical to quality assessment. A fundamental challenge involves deciphering intertwined chemical processes to characterize degradation patterns and their quantitative relationship with battery performance. Here we show that physics-informed machine learning can quantify and visualize temporally resolved losses concerning thermodynamics and kinetics using electric signals. The proposed method facilitates a non-destructive degradation pattern characterization, expediting temperature-adaptable predictions of entire lifetime trajectories, rather than end-of-life points. The verification speed is 25 times faster than capacity calibration tests with 95.1% prediction accuracy across temperatures. We attribute the predictability to interpreting statistical insights from material-agnostic featurization, elicited by a multistep charging profile, for degradation pattern decoupling. Sustainable management of defective prototypes before massive production is realized using statistically interpreted degradation information, demonstrating a 19.76 billion USD scrap material recycling market in China by 2060. Our findings offer new possibilities for transforming material prototypes into commercial products by shortening the quality verification time and favoring next-generation battery research and development sustainability.

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
* Prototype ternary nickel manganese cobalt lithium-ion batteries were cycled under controlled temperatureconditions (25, 35, 45, 55℃) under multi-step charging (0.33C to 3C, where 1C is 1.1A) and 1C constant discharge beyond EOL thresholds. We generate a unique battery prototype verification dataset spanning lifetimes of 480 to 1025 cycles (average lifetime of 775 with a standard deviation of 175 under EOL80 definition). Due to the data confidentiality agreement, we are unable to disclose the dataset publicly. If there is a need for subsequent research, please contact the author to obtain the data.


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
The entire experiment consists of three steps: 
* ChemicalProcessModel: Model multi-dimensional chemical processes by using early cycle and guiding sample data.
* DomainAdaptModel: Adapt these predictions to specific temperatures.
* DegradationTrajectoryModel: Use adapted chemical processes to predict the degradation in later cycles.

## 4.1 Chemical process prediction model considering initial manufacturing variability-ChemicalProcessModel
To allow the network to focus on relevant aspects of the voltage response matrix $x$ conditioned by the additional retirement condition information $cond$, we introduced the attention mechanism in both the encoder and decoder of the VAE. Here, we use the encoder as an example to illustrate.

The encoder network in the variational autoencoder is designed to process and compress input data into a latent space. It starts by taking the 21-dimensional battery voltage response feature matrix $x$ as main input and retirement condition matrix of the retired batteries $cond=[SOC,SOH]$ as conditional input. The condition input is first transformed into an embedding $C$, belonging to a larger latent space with 64-dimension. The conditional embedding $C$ is formulated as: 
$$C = \text{ReLU} \left( cond \cdot W_c^T + b_c \right)$$
where, $W_c$, $b_c$ are the condition embedding neural network weighting matrix and bias matrix, respectively. Here is the implementation:
```python
  # Embedding layer for conditional input (SOC + SOH)
    condition_input = Input(shape=(condition_dim,))
    condition_embedding = Dense(embedding_dim, activation='relu')(condition_input)
    condition_embedding_expanded = tf.expand_dims(condition_embedding, 2)
```
The main input matrix $x$, representing battery pulse voltage response features, is also transformed into this 64-dimensional latent space:
$$H = \text{ReLU} \left( x \cdot W_h^T + b_h \right)$$
where,  $W_h$, $b_h$ are the main input embedding neural network weighting matrix and bias matrix, respectively. Here is the implementation:
```python
  # Main input (21-dimensional features)
    x = Input(shape=(feature_dim,))
    # VAE Encoder
    h = Dense(intermediate_dim, activation='relu')(x)
    h_expanded = tf.expand_dims(h, 2)
```

Both $H$ and $C$ are then integrated via a cross-attention mechanism, allowing the network to focus on relevant aspects of the voltage response matrix $x$ conditioned by the additional retirement condition information $cond$:
$$AttenEncoder = \text{Attention}(H,C,C)$$ 
Here is the implementation:
```python
    # Cross-attention in Encoder
    attention_to_encode = MultiHeadAttention(num_heads, key_dim=embedding_dim)(
        query=h_expanded,
        key=condition_embedding_expanded,
        value=condition_embedding_expanded
    )
    attention_output_squeezed = tf.squeeze(attention_to_encode, 2)

    z_mean = Dense(latent_dim)(attention_output_squeezed)
    z_log_var = Dense(latent_dim)(attention_output_squeezed)
    z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])
    encoder = Model(inputs=[x, condition_input], outputs=[z_mean, z_log_var, z])
```
The primary function of the Decoder Network is to transform the sampled latent variable $z$ back into the original dataspace, reconstructing the input data or generating new data attended on the original or unseen retirement conditions. The first step in the decoder is a dense layer that transforms $z$ into an intermediate representation:
$$H^{'} = \text{ReLU} \left( z \cdot W_d^T + b_d \right)$$
```python
    # VAE Decoder
    z_input = Input(shape=(latent_dim,))
    decoder_h = Dense(intermediate_dim, activation='relu')
    decoder_mean = Dense(feature_dim, activation='sigmoid')
    h_decoded = decoder_h(z_input)
    h_decoded_expanded = tf.expand_dims(h_decoded, 2)
```
$H^{'}$ is then integrated via a cross-attention mechanism, allowing the network to focus on relevant aspects of the voltage response matrix $x$ conditioned by the additional retirement condition information $cond$:
$$AttenDecoder = \text{Attention}(H^{'},C^{'},C^{'})$$ 
```python
    # Cross-attention in Decoder
    attention_to_decoded = MultiHeadAttention(num_heads, key_dim=embedding_dim)(
        query=h_decoded_expanded,
        key=condition_embedding_expanded,
        value=condition_embedding_expanded
    )
    attention_output_decoded_squeezed = tf.squeeze(attention_to_decoded, 2)
    _x_decoded_mean = decoder_mean(attention_output_decoded_squeezed)
    decoder = Model(inputs=[z_input, condition_input], outputs=_x_decoded_mean)
```
With both the encoder and the decoder, the construction of the VAE model is
```python
    # VAE Model
    _, _, z = encoder([x, condition_input])
    vae_output = decoder([z, condition_input])
    vae = Model(inputs=[x, condition_input], outputs=vae_output)
```

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

