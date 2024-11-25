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

$$
\hat{\mathbf{F}} = f_\theta(U) = \left(f_\sigma^{(L)} \left(f_\theta^{(L)} \circ \cdots \circ f_\sigma^{(1)} \left(f_\theta^{(1)}\right)\right)\right)(U)
$$

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

The multi-source transfer based on $AT_{\text{score}}$ includes the following three steps. 

* First, we calculate aging rates $r$ for all target and source domains by using early-stage data, i.e., we set $\text{start} = 100$, $\text{end} = 200$, $n = 50$. After calculating aging rates for all features or aging curves, we obtain a target domain aging rate vector $r_{\text{target } 1\times N}$ and a source domain aging rate matrix $r_{\text{source } K\times N}$, where $K$ and $N$ are the number of source domains and the number of features, respectively.
```python
def gradient(feature_list,start,end,sample_size):
    mean_start = 0
    mean_end = 0
    for i in range(sample_size):
      mean_start = mean_start + feature_list[start+i]
    for i in range(sample_size):
      mean_end = mean_end + feature_list[end+i]
    mean_start = mean_start/sample_size
    mean_end = mean_end/sample_size
    grad = mean_end - mean_start
    rang = end-start
    return np.log(abs(grad/rang))

start45 = early_cycle_start#100
end45 = early_cycle_end#200
g25.append(gradient(pre25[i][:],start,end,sample_size))
g55.append(gradient(pre55[i][:],start,end,sample_size))
g35.append(gradient(real35[i][:],start,end,sample_size))
g45.append(gradient(real45[i][:],start45,end45,sample_size))
```

* Second, we calculate the transferability metric $AT_{\text{score}}$ and weight vector $W_{1\times K} = \{W_i\}$.

```python
#ATscore calculation
at25.append(gradient(real45[i][:],start45,end45,sample_size)/gradient(pre25[i][:],start,end,sample_size))
at55.append(gradient(real45[i][:],start45,end45,sample_size)/gradient(pre55[i][:],start,end,sample_size))

#weight calculation
w25 = abs(step25-1)+abs(step55-1)
w25 = abs(step55-1)/w25
w_at_25.append(w25)
```
* Third, we predict the late stage (cycles after 200) aging rate of the target domain ($r_{\text{target}}$) (shown in 4.2.3). Note that $AT_{\text{score}}^{\text{source } i \to \text{target}}$ and $W_i$ are obtained by both target and source domain early-stage data, which are used to measure the transferability from source domain to target domain based on their aging rate similarity. $r_{\text{source } i}$ is obtained from all accessible data in the source domain, consistent with our definition of the early-stage estimate problem.

Using the physics-informed transferability metric, we assign a weight vector $W_{1\times K} = \{W_i\}$ (where $K$ is the number of source domains, $W_i$ is the ensemble weight for the $i$-th source domain) to source domains to quantify the contributions when predicting the chemical process of the target domain. The $W_i$ is defined as:

$$
W_i = \left( \left| AT_{\text{score}}^{\text{source } i \to \text{target}} - 1 \right| \cdot \left( \sum_{j=1}^K \frac{1}{\left| AT_{\text{score}}^{\text{source } j \to \text{target}} - 1 \right|} \right)^{-1} \right) \tag{10}
$$

where, $AT_{\text{score}}^{\text{source } i \to \text{target}}$ is the $AT_{\text{score}}$ from the $i$-th source domain to the target domain. This mechanism ensures the source domain with better transferability has a higher weight, effectively quantifying the contribution of each source domain to the prediction of the target domain. From Equation (6) and Equation (10), we can obtain the aging rate of the target domain:

$$
r_{\text{target}} = \sum_{i=1}^K W_i \cdot AT_{\text{score}}^{\text{source } i \to \text{target}} \cdot r_{\text{source } i} \tag{11}
$$



See the Methods section of the paper for more details.


### 4.2.3 Chain of degradation

### Chain of Degradation

Battery chemical process degradation is continuous, which we call the "Chain of Degradation". We have predicted the $r_{\text{target}}$ aging rates of each feature in the target domain, which can be further used to predict the chemical process. Therefore, when using aging rates $r_{\text{target}}$ to calculate each target feature vector $F_{\text{(C×m)×1}}$ in the feature matrix $F_{\text{(C×m)×N}}$, the $i$-th cycle target feature vector $F_{\text{target}}^i$ should be based on $F_{\text{target}}^{i-1}$ and $r^{i-1}$:

$$
F_{\text{target}}^i = F_{\text{target}}^{i-1} + \sum_{j=i}^K W_j \cdot A_{\text{score}}^{\text{source } j \to \text{target}} \cdot r_{\text{source } j}^{i-1} \tag{12}
$$

where $F_{\text{target}}^i$ is the feature value of the target domain in the $i$-th cycle, $r_{\text{source } j}^{i-1}$ is the aging rate of source domain $j$ at the $(i-1)$-th cycle.

We concatenate the $N$ feature vectors $F_{\text{(C×m)×1}}$ to get the feature matrix $F_{\text{(C×m)×N}}$. Since our chemical process prediction for each step is based on the result of the previous step, we can track the accumulation of degradation in the aging process and thus it is robust against noise.
Here is the implementation:
```python

      #################################################################################################
      # AT method:
      # Input: at25_to_45, at55_to_45, w_at_25, pre25, pre55 (these are from this round), last45, last55, last25 (these are from the previous round).
      # Output: pre45, stored in feature_1 (replaces the original model1 output).
      # Renew: last45 = pre45, last25 = pre25, last55 = pre55
      #################################################################################################

      if(batch>early_cycle_end):
        if(batch==early_cycle_end):
            pred_feature.append(last45)
        for i in range(42):

          step1 = w_at_25[i]*(pre25[i][0]-last25[i][0])*at25[i]
          step2 = (1-w_at_25[i])*(pre55[i][0]-last55[i][0])*at55[i]
          feature_45[i] =  last45[i] + w_at_25[i]*(pre25[i][0]-last25[i][0])*at25[i] + (1-w_at_25[i])*(pre55[i][0]-last55[i][0])*at55[i]
          last45[i] = feature_1[i]
          last25[i][0] = pre25[i][0]
          last55[i][0] = pre55[i][0]
        pred_feature.append(feature_45)

```

## 4.3 Battery degradation trajectory model
We have successfully predicted the battery chemical process. It is assumed that the chemical process of the battery deterministically affects the aging process, we therefore use the predicted chemical process to predict the battery degradation curve. The battery degradation trajectory model learns a composition of $L$ intermediate mappings:

$$
\hat{\mathbf{D}} = f_\theta(\hat{\mathbf{F}}) = \left(f_\sigma^{(L)} \left(f_\theta^{(L)} \circ \cdots \circ f_\sigma^{(1)} \left(f_\theta^{(1)}\right)\right)\right)(\hat{\mathbf{F}})
$$

where $L = 3$ in this work. $\hat{\mathbf{D}} $ is predicted battery degradation trajectories, $\theta = \{\theta^{(1)}, \theta^{(2)}, \theta^{(3)}\}$ is the collection of network parameters for each layer, $\hat{\mathbf{F}}$ is the predicted battery chemical process feature matrix, and $f_\theta(\hat{\mathbf{F}})$ is a neural network predictor. All layers are fully connected. The activation function used is Leaky ReLU (leaky rectified linear unit), denoted as $f_\sigma$. 

Here is the implementation:DegradationTrajectory
```python
   class DegradationTrajectoryModel(nn.Module):
    def __init__(self):
        super(DegradationTrajectoryModel, self).__init__()
        self.fc1 = nn.Linear(53, 32)
        self.fc2 = nn.Linear(32, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x
```
# 5. Access
Access the raw data and processed features [here]((https://github.com/terencetaothucb/TBSI-Sunwoda-Battery-Dataset)) under the [MIT licence](https://github.com/terencetaothucb/Pulse-Voltage-Response-Generation/blob/main/LICENSE). Correspondence to [Terence (Shengyu) Tao](terencetaotbsi@gmail.com) and CC Prof. [Xuan Zhang](xuanzhang@sz.tsinghua.edu.cn) and [Guangmin Zhou](guangminzhou@sz.tsinghua.edu.cn) when you use, or have any inquiries.
# 6. Acknowledgements
[Terence (Shengyu) Tao](mailto:terencetaotbsi@gmail.com) and [Zixi Zhao](zhaozx23@mails.tsinghua.edu.cn)  at Tsinghua Berkeley Shenzhen Institute designed the model and algorithms, developed and tested the experiments, uploaded the model and experimental code, revised the testing experiment plan, and wrote this instruction document based on supplementary materials.  

