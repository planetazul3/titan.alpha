# 🚀 Suite de Prompts Unificada — Arquitectura Multi-Expert de Trading

> Guía completa para construir un sistema de Deep Learning modular para contratos de opciones binarias en Deriv.com

---

## 📖 Resumen Ejecutivo

Este documento unifica tres versiones de prompts en una suite coherente y optimizada. El sistema implementa:

- **Arquitectura Micro-Modular**: Archivos pequeños (40-200 líneas) con responsabilidad única
- **Tres Expertos Especializados**: Temporal (LSTM), Espacial (CNN), Volatilidad (Autoencoder)
- **Sistema de Ejecución Inteligente**: Clasificación de señales REAL/SHADOW/IGNORE
- **Configuración Robusta**: Pydantic con validación estricta y fail-fast

---

## 📁 Estructura del Proyecto

```
project/
├── config/
│   ├── __init__.py
│   ├── settings.py          # BaseSettings Pydantic con validación
│   └── constants.py          # Enums y constantes globales
├── data/
│   ├── __init__.py
│   ├── normalizers.py        # Normalización: log-returns, z-score, min-max
│   ├── indicators.py         # RSI, Bollinger Bands, ATR (NumPy puro)
│   ├── preprocessor.py       # Orquestador de preprocesamiento
│   ├── dataset.py            # PyTorch Dataset (lazy loading)
│   └── loader.py             # DataLoader factory con seeding
├── models/
│   ├── __init__.py
│   ├── attention.py          # Additive + Scaled Dot-Product Attention
│   ├── blocks.py             # BiLSTM, Conv1D, MLP reutilizables
│   ├── temporal.py           # Expert temporal (Rise/Fall)
│   ├── spatial.py            # Expert espacial (Barriers/Runs)
│   ├── volatility.py         # Expert volatilidad (Ranges)
│   ├── fusion.py             # Capa de fusión multi-expert
│   ├── heads.py              # Output heads por contrato
│   └── core.py               # DerivOmniModel integrador
├── execution/
│   ├── __init__.py
│   ├── signals.py            # Dataclasses para señales
│   ├── filters.py            # Filtros de probabilidad
│   ├── shadow_logger.py      # Logging de operaciones simuladas
│   └── decision.py           # Motor de decisión principal
├── utils/
│   ├── __init__.py
│   ├── device.py             # Resolución GPU/CPU/MPS
│   └── seed.py               # Seeding global para reproducibilidad
├── .env                       # Variables de entorno
└── requirements.txt           # Dependencias
```

---

## 🔄 Orden de Implementación

| Fase | Módulos | Dependencias |
|------|---------|--------------|
| **1. Fundamentos** | constants, device, seed, settings | Ninguna |
| **2. Data Pipeline** | normalizers → indicators → preprocessor → dataset → loader | Fase 1 |
| **3. Model Blocks** | attention → blocks | Fase 1 |
| **4. Experts** | temporal → spatial → volatility | Fase 3 |
| **5. Integration** | fusion → heads → core | Fase 4 |
| **6. Execution** | signals → filters → shadow_logger → decision | Fase 5 |

---

# 🎯 FASE 1: Fundamentos y Configuración

## Prompt 1.1 — Constantes y Utilidades Base

```markdown
Act as a Senior Python Architect specialized in trading systems and PyTorch.

Task: Create the foundational utility modules that all other components will import.

---

### File 1: `config/constants.py`

**Purpose:** Centralize static definitions to avoid magic numbers and scattered strings.

**Required Content:**
1. `CONTRACT_TYPES`: Enum with values `RISE_FALL`, `TOUCH_NO_TOUCH`, `STAYS_BETWEEN`
2. `SIGNAL_TYPES`: Enum with values `REAL_TRADE`, `SHADOW_TRADE`, `IGNORE`
3. Numeric constants:
   - `MIN_SEQUENCE_LENGTH = 16`
   - `MAX_SEQUENCE_LENGTH = 2000`
   - `DEFAULT_SEED = 42`

**Constraints:** 
- Use UPPERCASE for all constants
- Include docstrings explaining each constant's purpose
- < 80 lines

---

### File 2: `utils/device.py`

**Purpose:** Manage hardware acceleration (GPU/MPS/CPU).

**Required Content:**
1. Function `resolve_device(preference: Literal['cpu', 'cuda', 'mps', 'auto']) -> torch.device`
   - Check CUDA (Nvidia) availability first
   - Check MPS (Apple Silicon) availability second
   - Fallback to CPU if necessary
   - Include simple logging indicating selected device
2. Function `get_device_info() -> Dict[str, Any]` returning device capabilities

**Constraints:** < 60 lines

---

### File 3: `utils/seed.py`

**Purpose:** Guarantee experiment reproducibility.

**Required Content:**
1. Function `set_global_seed(seed: int) -> None`
   - Seed: `random`, `numpy`, `torch`, `torch.cuda`
   - Set `torch.backends.cudnn.deterministic = True`
   - Set `torch.backends.cudnn.benchmark = False`
2. Docstring explaining reproducibility caveats with multi-threading

**Constraints:** < 50 lines

---

Output: Complete code for all 3 files with proper type hints and docstrings.
```

---

## Prompt 1.2 — Sistema de Configuración Robusto

```markdown
Act as a Configuration Management Specialist for ML systems.

Task: Create a robust, typed configuration system using Pydantic that fails fast if required variables are missing.

---

### File: `config/settings.py`

**Requirements:**

1. **Import dependencies:**
   - `MIN_SEQUENCE_LENGTH` from `config.constants`
   - `resolve_device` from `utils.device`

2. **Nested Pydantic models (exact names required):**

   ```python
   class Trading(BaseModel):
       symbol: str                    # e.g., "R_100"
       timeframe: str = "1m"
       stake_amount: float

   class Thresholds(BaseModel):
       confidence_threshold_high: float  # >= this → REAL_TRADE
       learning_threshold_min: float     # >= this → SHADOW_TRADE
       learning_threshold_max: float     # < this → SHADOW_TRADE
       # Validator: min < max < high

   class ModelHyperparams(BaseModel):
       learning_rate: float
       batch_size: int
       dropout_rate: float = 0.1
       lstm_hidden_size: int
       cnn_filters: int
       latent_dim: int

   class DataShapes(BaseModel):
       sequence_length_ticks: int   # >= MIN_SEQUENCE_LENGTH
       sequence_length_candles: int # >= MIN_SEQUENCE_LENGTH
   ```

3. **Main Settings class:**
   - Inherits from `BaseSettings`
   - Contains all nested models above
   - General fields: `environment`, `seed`, `device_preference`
   - Read from `.env` file with `env_nested_delimiter='__'`

4. **Validation rules:**
   - `learning_threshold_min < learning_threshold_max < confidence_threshold_high`
   - `sequence lengths >= MIN_SEQUENCE_LENGTH`
   - `learning_rate > 0`, `batch_size > 0`, `0 <= dropout_rate < 1`

5. **Helper methods:**
   - `get_device() -> torch.device`
   - `validate_thresholds() -> bool`

6. **Factory function:**
   - `load_settings() -> Settings` with runtime validation

---

### File: `.env` (example)

```ini
# Trading
TRADING__SYMBOL=R_100
TRADING__TIMEFRAME=1m
TRADING__STAKE_AMOUNT=1.0

# Thresholds
THRESHOLDS__CONFIDENCE_THRESHOLD_HIGH=0.85
THRESHOLDS__LEARNING_THRESHOLD_MIN=0.40
THRESHOLDS__LEARNING_THRESHOLD_MAX=0.84

# Hyperparameters
HYPERPARAMS__LEARNING_RATE=0.0005
HYPERPARAMS__BATCH_SIZE=128
HYPERPARAMS__DROPOUT_RATE=0.2
HYPERPARAMS__LSTM_HIDDEN_SIZE=256
HYPERPARAMS__CNN_FILTERS=64
HYPERPARAMS__LATENT_DIM=16

# Data shapes
DATA_SHAPES__SEQUENCE_LENGTH_TICKS=1000
DATA_SHAPES__SEQUENCE_LENGTH_CANDLES=200

# General
ENVIRONMENT=development
SEED=42
DEVICE_PREFERENCE=auto
```

---

Output: Complete `config/settings.py` and example `.env` file.
```

---

# 🎯 FASE 2: Pipeline de Datos

## Prompt 2.1 — Normalizadores Numéricos

```markdown
Act as a Quantitative Data Engineer.

Task: Create `data/normalizers.py` with pure NumPy mathematical functions for financial data normalization.

---

**Functions to implement:**

### 1. `log_returns(prices: np.ndarray, fill_first: bool = True) -> np.ndarray`
- Formula: $\ln(p_t / p_{t-1})$
- Handle first value (NaN → 0 if fill_first=True)
- Docstring with mathematical formula

### 2. `z_score_normalize(values: np.ndarray, window: Optional[int] = None, epsilon: float = 1e-8) -> np.ndarray`
- If `window=None`: Use global mean/std
- If `window=int`: Rolling z-score
- Handle division by zero with epsilon
- Formula: $(x - \mu) / (\sigma + \epsilon)$

### 3. `min_max_normalize(values: np.ndarray, feature_range: Tuple[float, float] = (0, 1)) -> np.ndarray`
- Standard min-max scaling to specified range
- Formula: $\frac{x - x_{min}}{x_{max} - x_{min}} \cdot (max - min) + min$

### 4. `robust_scale(values: np.ndarray, quantile_range: Tuple[float, float] = (25, 75)) -> np.ndarray`
- Use IQR instead of standard deviation (robust to outliers)
- Formula: $(x - median) / IQR$

---

**Constraints:**
- Pure NumPy (no PyTorch, no pandas)
- Vectorized operations (no for loops)
- Each function < 30 lines
- Complete type hints
- Docstrings with formulas and usage examples

Output: Complete file < 150 lines
```

---

## Prompt 2.2 — Indicadores Técnicos (NumPy Puro)

```markdown
Act as a High-Performance Technical Analysis Engineer.

Task: Create `data/indicators.py` with financial indicators using pure NumPy (no pandas-ta, no ta-lib).

---

**Functions to implement:**

### Helper: `rolling_window(a: np.ndarray, window: int) -> np.ndarray`
- Efficient sliding window view using `np.lib.stride_tricks`
- Handle edge cases properly

### 1. `rsi(prices: np.ndarray, period: int = 14) -> np.ndarray`
- Standard RSI (0-100 scale)
- Use Wilder's smoothing method (EMA)
- Handle initial NaN values

### 2. `bollinger_bands(prices: np.ndarray, period: int = 20, std_dev: float = 2.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]`
- Returns: (upper, middle, lower)
- Middle = SMA(period)
- Bands = middle ± (std_dev × rolling_std)

### 3. `atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray`
- Average True Range with Wilder smoothing
- TR = max(high-low, |high-prev_close|, |low-prev_close|)

### 4. `bollinger_bandwidth(upper: np.ndarray, lower: np.ndarray, middle: np.ndarray) -> np.ndarray`
- Formula: (upper - lower) / middle
- Useful as volatility feature

### 5. `bollinger_percent_b(prices: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> np.ndarray`
- Formula: (price - lower) / (upper - lower)
- Position within bands (0-1)

---

**Constraints:**
- Maximum efficiency (vectorized)
- Proper NaN handling at series start
- < 200 lines total
- Each function < 40 lines
- Docstrings with mathematical formulas

Output: Complete file
```

---

## Prompt 2.3 — Orquestador de Preprocesamiento

```markdown
Act as a Data Pipeline Architect.

Task: Create `data/preprocessor.py` that orchestrates normalizers and indicators to transform raw data into ML-ready tensors.

---

**Classes to implement:**

### 1. `TickPreprocessor`

```python
class TickPreprocessor:
    def __init__(self, settings: Settings):
        self.target_length = settings.data_shapes.sequence_length_ticks
    
    def process(self, ticks: np.ndarray) -> np.ndarray:
        """
        Transform raw tick prices to normalized sequence.
        
        Steps:
        1. Apply log_returns
        2. Apply z_score_normalize
        3. Pad/truncate to target_length
        
        Returns: shape (target_length,), dtype float32
        """
```

### 2. `CandlePreprocessor`

```python
class CandlePreprocessor:
    def __init__(self, settings: Settings):
        self.target_length = settings.data_shapes.sequence_length_candles
    
    def process(self, ohlcv: np.ndarray) -> np.ndarray:
        """
        Transform OHLCV data with technical indicators.
        
        Input: (N, 6) → [Open, High, Low, Close, Volume, Time]
        
        Steps:
        1. Normalize OHLC with log_returns
        2. Normalize Volume with z_score
        3. Compute: RSI, BB_width, BB_pct, ATR
        4. Concatenate all features
        
        Returns: shape (target_length, 10), dtype float32
        Features: [O_norm, H_norm, L_norm, C_norm, V_norm, Time_norm, RSI, BB_width, BB_pct, ATR]
        """
```

### 3. `VolatilityMetricsExtractor`

```python
class VolatilityMetricsExtractor:
    def extract(self, candles: np.ndarray) -> np.ndarray:
        """
        Extract aggregated volatility metrics for autoencoder input.
        
        Returns: shape (n_features,) with:
        - realized_volatility (std of log returns)
        - atr_mean
        - rsi_std  
        - bb_width_mean
        """
```

---

**Constraints:**
- Import from `data.normalizers` and `data.indicators`
- Each class < 60 lines
- No look-ahead bias (use only past data)
- Proper docstrings explaining output shapes

Output: Complete file < 180 lines
```

---

## Prompt 2.4 — PyTorch Dataset

```markdown
Act as a PyTorch Data Engineer.

Task: Create `data/dataset.py` with `DerivDataset` class.

---

**Specifications:**

```python
class DerivDataset(torch.utils.data.Dataset):
    def __init__(
        self, 
        data_source: Any,  # Path or data handler
        settings: Settings,
        mode: Literal['train', 'eval'] = 'train'
    ):
        """
        Initialize dataset with lazy loading.
        
        - Store preprocessor instances
        - Index available samples
        - Do NOT load all data into memory
        """
    
    def __len__(self) -> int:
        """Return total number of samples."""
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Load and process single sample on-demand.
        
        Returns:
        {
            'ticks': Tensor[float32],       # (seq_len_ticks,)
            'candles': Tensor[float32],     # (seq_len_candles, 10)
            'vol_metrics': Tensor[float32], # (n_features,)
            'target': Tensor[float32/int64] # scalar or (1,)
        }
        """
```

---

**Requirements:**
- Lazy loading (process on `__getitem__`, not `__init__`)
- Use preprocessors from `data.preprocessor`
- Handle missing/corrupt data gracefully
- Ensure `dtype=torch.float32` for features, `torch.int64` for labels if classification
- No hardcoded lengths (use settings)
- < 120 lines

Output: Complete file
```

---

## Prompt 2.5 — DataLoader Factory

```markdown
Act as an ML Ops Engineer.

Task: Create `data/loader.py` with DataLoader factory function.

---

**Function signature:**

```python
def create_dataloaders(
    train_data: Any,
    val_data: Any,
    settings: Settings
) -> Tuple[DataLoader, DataLoader]:
    """
    Create configured DataLoaders for training and validation.
    
    Configuration:
    - batch_size: from settings.hyperparams
    - num_workers: auto-detect from os.cpu_count()
    - pin_memory: True if CUDA available
    - shuffle: True for train, False for val
    - worker_init_fn: deterministic seeding per worker
    - drop_last: True for train (consistent batch sizes)
    
    Returns: (train_loader, val_loader)
    """
```

---

**Helper function:**

```python
def worker_init_fn(worker_id: int) -> None:
    """
    Initialize each DataLoader worker with deterministic but different seeds.
    Ensures reproducibility across runs.
    """
```

---

**Constraints:**
- Import `DerivDataset` from `data.dataset`
- < 80 lines
- Clear docstrings

Output: Complete file
```

---

# 🎯 FASE 3: Arquitectura de Modelos Neuronales

## Prompt 3.1 — Mecanismos de Atención

```markdown
Act as a Deep Learning Researcher specializing in attention mechanisms.

Task: Create `models/attention.py` with modular, reusable attention modules.

---

**Classes to implement:**

### 1. `AdditiveAttention(nn.Module)`

```python
class AdditiveAttention(nn.Module):
    """
    Bahdanau-style additive attention for sequential data.
    
    Reference: Bahdanau et al. (2014) "Neural Machine Translation by 
               Jointly Learning to Align and Translate"
    
    Input: (batch, seq_len, hidden_dim)
    Output: (context_vector, attention_weights)
            context_vector: (batch, hidden_dim)
            attention_weights: (batch, seq_len)
    """
    
    def __init__(self, hidden_dim: int):
        # Linear projections for Query, Key, Score
    
    def forward(self, encoder_outputs: torch.Tensor, mask: Optional[torch.Tensor] = None):
        # Compute attention scores and context vector
```

### 2. `ScaledDotProductAttention(nn.Module)`

```python
class ScaledDotProductAttention(nn.Module):
    """
    Transformer-style scaled dot-product attention.
    
    Formula: softmax(QK^T / sqrt(d_k)) @ V
    
    Input: Q, K, V tensors
    Output: (output, attention_weights)
    """
    
    def __init__(self, temperature: Optional[float] = None):
        # temperature = sqrt(d_k) if not specified
    
    def forward(self, q, k, v, mask=None):
        # Compute scaled attention
```

---

**Constraints:**
- Device-agnostic (use `input.device`)
- Efficient memory usage
- Support optional masking
- < 100 lines total
- Complete type hints and docstrings

Output: Complete file
```

---

## Prompt 3.2 — Bloques Reutilizables

```markdown
Act as a Neural Network Architecture Designer.

Task: Create `models/blocks.py` with composable building blocks.

---

**Classes to implement:**

### 1. `BiLSTMBlock(nn.Module)`

```python
class BiLSTMBlock(nn.Module):
    """
    Wrapper around nn.LSTM with bidirectional=True.
    
    Features:
    - Automatic hidden state initialization
    - Returns full sequence of outputs
    - Proper packing/unpacking for variable lengths (optional)
    """
    
    def __init__(
        self, 
        input_size: int, 
        hidden_size: int, 
        num_layers: int = 2,
        dropout: float = 0.1
    ):
        # LSTM with bidirectional=True
    
    def forward(self, x: torch.Tensor, lengths: Optional[torch.Tensor] = None):
        # Returns: (outputs, (h_n, c_n))
        # outputs shape: (batch, seq_len, hidden_size * 2)
```

### 2. `Conv1DBlock(nn.Module)`

```python
class Conv1DBlock(nn.Module):
    """
    Single convolutional block: Conv1d → BatchNorm1d → Activation → Pool (optional)
    
    Uses SiLU (Swish) activation by default.
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: str = 'same',
        activation: str = 'silu',
        pool_size: Optional[int] = None
    ):
```

### 3. `MLPBlock(nn.Module)`

```python
class MLPBlock(nn.Module):
    """
    Flexible Multi-Layer Perceptron.
    
    Features:
    - Configurable layer sizes
    - Optional dropout between layers
    - Optional layer normalization
    - Activation after each layer except last
    """
    
    def __init__(
        self,
        layer_sizes: List[int],  # e.g., [input, 512, 256, output]
        activation: str = 'silu',
        dropout: float = 0.0,
        use_layer_norm: bool = False
    ):
```

---

**Constraints:**
- Proper weight initialization (Xavier/Kaiming)
- Device-agnostic
- Complete type hints
- < 180 lines total

Output: Complete file
```

---

## Prompt 3.3 — Experto Temporal (Rise/Fall)

```markdown
Act as a Time-Series Deep Learning Expert.

Task: Create `models/temporal.py` with `TemporalExpert` specialized for Rise/Fall binary contracts.

---

**Specifications:**

```python
class TemporalExpert(nn.Module):
    """
    Temporal expert for directional prediction (Rise/Fall contracts).
    
    Architecture:
    1. BiLSTM to capture temporal dependencies
    2. Additive Attention to focus on key candles
    3. MLP projection to embedding space
    
    Purpose: Capture medium-term directional market intention.
    
    Why BiLSTM + Attention:
    - Bidirectional: context from both past and future helps identify patterns
    - Attention: weighs important candles (reversals, breakouts) more heavily
    """
    
    def __init__(self, settings: Settings, embedding_dim: int = 64):
        """
        Args:
            settings: Configuration with hyperparameters
            embedding_dim: Output embedding dimension
        """
        # Use settings.hyperparams.lstm_hidden_size
        # Use settings.data_shapes.sequence_length_candles for input size reference
    
    def forward(self, candles: torch.Tensor) -> torch.Tensor:
        """
        Args:
            candles: (batch, seq_len, features) normalized candle data
        
        Returns:
            embedding: (batch, embedding_dim) temporal representation
        """
```

---

**Implementation details:**
- Import `BiLSTMBlock` from `models.blocks`
- Import `AdditiveAttention` from `models.attention`
- MLP: Linear → SiLU → Dropout → Linear → SiLU
- NO Sigmoid/Softmax at output (return raw embedding)
- Use `candles.device` for device agnosticism
- Apply proper weight initialization

**Constraints:**
- < 80 lines
- No hardcoded dimensions (use settings)
- Clear docstrings

Output: Complete file
```

---

## Prompt 3.4 — Experto Espacial (Barriers/Runs)

```markdown
Act as a Signal Processing Neural Network Designer.

Task: Create `models/spatial.py` with `SpatialExpert` for tick-series geometry analysis.

---

**Specifications:**

```python
class SpatialExpert(nn.Module):
    """
    Spatial expert for barrier/touch/run contracts.
    
    Architecture: Pyramidal 1D-CNN with increasing receptive fields
    
    Block1: kernel=3 (micro patterns, ~30ms)
    Block2: kernel=5 (medium patterns, ~50ms)  
    Block3: kernel=15 (macro patterns, ~150ms)
    
    Purpose: Analyze "roughness" and geometry of price curves.
    Larger kernels capture momentum, smaller capture noise patterns.
    """
    
    def __init__(self, settings: Settings, embedding_dim: int = 64):
        """
        Args:
            settings: Configuration with cnn_filters
            embedding_dim: Output embedding dimension
        """
        # Base channels = settings.hyperparams.cnn_filters
        # Expand: cnn_filters → cnn_filters*2 → cnn_filters*4
    
    def forward(self, ticks: torch.Tensor) -> torch.Tensor:
        """
        Args:
            ticks: (batch, 1, seq_len) normalized tick series
        
        Returns:
            embedding: (batch, embedding_dim) spatial representation
        """
```

---

**Implementation details:**
- Import `Conv1DBlock` from `models.blocks`
- Use `nn.AdaptiveAvgPool1d(1)` to collapse temporal dimension
- Flatten + Linear projection to embedding_dim
- Include shape assertions with helpful error messages
- Document receptive field calculations in comments

**Constraints:**
- < 70 lines
- Careful stride/padding to control border effects
- Clear comments on architectural choices

Output: Complete file
```

---

## Prompt 3.5 — Experto de Volatilidad (Autoencoder)

```markdown
Act as an Unsupervised Learning Specialist.

Task: Create `models/volatility.py` with `VolatilityExpert` autoencoder for range contracts.

---

**Specifications:**

```python
class VolatilityExpert(nn.Module):
    """
    Volatility expert using autoencoder architecture.
    
    Purpose: Learn compressed representation of volatility regime.
    High reconstruction error → unusual volatility → avoid range contracts.
    
    Architecture:
    - Encoder: input_dim → hidden → latent_dim (bottleneck)
    - Decoder: latent_dim → hidden → input_dim (reconstruction)
    
    Only the latent vector is used for fusion; decoder for training.
    """
    
    def __init__(
        self, 
        input_dim: int,  # Number of volatility metrics
        settings: Settings,
        hidden_dim: int = 32
    ):
        """
        Args:
            input_dim: Dimension of volatility metrics vector
            settings: Configuration with latent_dim
            hidden_dim: Intermediate layer size
        """
        # latent_dim = settings.hyperparams.latent_dim
        # Symmetric encoder/decoder architecture
    
    def forward(self, vol_metrics: torch.Tensor) -> torch.Tensor:
        """
        Returns only the latent embedding.
        
        Args:
            vol_metrics: (batch, input_dim) volatility metrics
        
        Returns:
            latent: (batch, latent_dim) compressed representation
        """
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input to latent space."""
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent vector to reconstruction."""
    
    def reconstruct(self, x: torch.Tensor) -> torch.Tensor:
        """Full encode → decode pass."""
    
    def reconstruction_error(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute per-sample MSE reconstruction error.
        
        Returns: (batch,) tensor of error values
        
        Interpretation:
        - Low error: Normal volatility regime
        - High error: Anomalous market conditions → avoid range contracts
        """
```

---

**Implementation details:**
- Use LeakyReLU or SiLU for activations
- Add BatchNorm between layers
- Symmetric architecture (encoder mirrors decoder)
- Include `reconstruction_loss(input, recon)` static helper method

**Constraints:**
- < 90 lines
- Clear docstring on anomaly interpretation
- Type hints throughout

Output: Complete file
```

---

## Prompt 3.6 — Capa de Fusión

```markdown
Act as a Multi-Modal Fusion Architect.

Task: Create `models/fusion.py` for expert embedding fusion.

---

**Specifications:**

```python
class ExpertFusion(nn.Module):
    """
    Fuses embeddings from all three experts into unified market context.
    
    Strategy: Concatenation + MLP mixing
    
    Alternative strategies (documented for future exploration):
    - Attention-weighted fusion
    - Gating mechanism
    - Bilinear fusion
    """
    
    def __init__(
        self,
        temporal_dim: int = 64,
        spatial_dim: int = 64,
        volatility_dim: int = 16,  # latent_dim from settings
        output_dim: int = 256
    ):
        """
        Total input: temporal_dim + spatial_dim + volatility_dim
        Output: Fused context vector of size output_dim
        """
    
    def forward(
        self,
        emb_temporal: torch.Tensor,
        emb_spatial: torch.Tensor,
        emb_volatility: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            emb_temporal: (batch, temporal_dim)
            emb_spatial: (batch, spatial_dim)
            emb_volatility: (batch, volatility_dim)
        
        Returns:
            fused: (batch, output_dim) global market context
        """
```

---

**Implementation:**
- Concatenate all embeddings
- Pass through MLP: concat → Linear(512) → BatchNorm → ReLU → Dropout → Linear(output_dim) → ReLU

**Constraints:**
- < 60 lines
- Modular (easy to swap fusion strategy)
- Clear docstring on fusion rationale

Output: Complete file
```

---

## Prompt 3.7 — Cabezales de Salida

```markdown
Act as a Classification Architecture Specialist.

Task: Create `models/heads.py` with contract-specific output heads.

---

**Specifications:**

```python
class ContractHead(nn.Module):
    """
    Generic output head for a single contract type.
    
    Returns raw logit (no Sigmoid) for numerical stability with BCEWithLogitsLoss.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: Optional[int] = None,
        dropout: float = 0.1
    ):
        """
        Args:
            input_dim: Fused context dimension
            hidden_dim: Optional intermediate layer (None = direct projection)
            dropout: Dropout rate before output
        """
    
    def forward(self, context: torch.Tensor) -> torch.Tensor:
        """
        Args:
            context: (batch, input_dim) fused market context
        
        Returns:
            logit: (batch, 1) raw logit for this contract
        """


def create_contract_heads(
    input_dim: int,
    settings: Settings
) -> nn.ModuleDict:
    """
    Factory function to create all contract heads.
    
    Returns:
        nn.ModuleDict with keys: 'rise_fall', 'touch', 'range'
    """
```

---

**Constraints:**
- < 70 lines total
- Clear docstring explaining why logits (not probs)
- Type hints

Output: Complete file
```

---

## Prompt 3.8 — OmniModel Core

```markdown
Act as a Systems Integration Engineer.

Task: Create `models/core.py` with `DerivOmniModel` that composes all experts.

---

**Specifications:**

```python
class DerivOmniModel(nn.Module):
    """
    Unified multi-expert model for Deriv binary options trading.
    
    Architecture:
    ┌─────────────┐   ┌─────────────┐   ┌─────────────────┐
    │   Candles   │   │    Ticks    │   │  Vol Metrics    │
    └──────┬──────┘   └──────┬──────┘   └────────┬────────┘
           │                 │                    │
           ▼                 ▼                    ▼
    ┌──────────────┐ ┌──────────────┐ ┌──────────────────┐
    │TemporalExpert│ │SpatialExpert │ │VolatilityExpert  │
    └──────┬───────┘ └──────┬───────┘ └────────┬─────────┘
           │                │                   │
           └────────────────┼───────────────────┘
                            ▼
                    ┌──────────────┐
                    │ ExpertFusion │
                    └──────┬───────┘
                           │
           ┌───────────────┼───────────────┐
           ▼               ▼               ▼
    ┌────────────┐ ┌────────────┐ ┌────────────┐
    │ Rise/Fall  │ │   Touch    │ │   Range    │
    │   Head     │ │   Head     │ │   Head     │
    └────────────┘ └────────────┘ └────────────┘
    """
    
    def __init__(self, settings: Settings):
        """Initialize all sub-modules from settings."""
    
    def forward(
        self,
        ticks: torch.Tensor,
        candles: torch.Tensor,
        vol_metrics: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Full forward pass through all experts and heads.
        
        Args:
            ticks: (batch, 1, seq_len_ticks)
            candles: (batch, seq_len_candles, features)
            vol_metrics: (batch, n_vol_features)
        
        Returns:
            Dict with keys:
            - 'rise_fall_logit': (batch, 1)
            - 'touch_logit': (batch, 1)
            - 'range_logit': (batch, 1)
        """
    
    def predict_probs(
        self,
        ticks: torch.Tensor,
        candles: torch.Tensor,
        vol_metrics: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with Sigmoid applied for inference.
        
        Returns probabilities instead of logits.
        """
    
    def count_parameters(self) -> int:
        """Return total trainable parameters for diagnostics."""
    
    def get_volatility_anomaly_score(
        self, 
        vol_metrics: torch.Tensor
    ) -> torch.Tensor:
        """
        Get reconstruction error from volatility expert.
        Useful for risk management decisions.
        """
```

---

**Constraints:**
- < 100 lines (leverages all modular components)
- Clean forward pass with clear data flow
- Type hints and comprehensive docstrings
- No I/O in this module (pure computation)

Output: Complete file
```

---

# 🎯 FASE 4: Lógica de Ejecución

## Prompt 4.1 — Definición de Señales

```markdown
Act as a Trading Systems Data Modeler.

Task: Create `execution/signals.py` with signal data structures.

---

**Specifications:**

```python
from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any
from datetime import datetime
from config.constants import SIGNAL_TYPES, CONTRACT_TYPES

@dataclass
class TradeSignal:
    """
    Base signal structure for trade opportunities.
    """
    signal_type: SIGNAL_TYPES       # REAL_TRADE, SHADOW_TRADE, IGNORE
    contract_type: CONTRACT_TYPES   # RISE_FALL, TOUCH_NO_TOUCH, STAYS_BETWEEN
    direction: Optional[str]        # 'CALL', 'PUT', None
    probability: float              # Model confidence (0-1)
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary for JSON export."""
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'TradeSignal':
        """Deserialize from dictionary."""


@dataclass
class ShadowTrade(TradeSignal):
    """
    Extended signal for paper trading with outcome tracking.
    """
    trade_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    entry_price: Optional[float] = None
    exit_price: Optional[float] = None
    outcome: Optional[bool] = None  # True = win, False = loss, None = pending
    pnl: Optional[float] = None     # Calculated P&L if outcome known
    
    def to_record(self) -> Dict[str, Any]:
        """Convert to flat record for logging/training."""
    
    def update_outcome(
        self, 
        outcome: bool, 
        exit_price: float, 
        stake: float
    ) -> None:
        """Update trade with final outcome."""
```

---

**Constraints:**
- Use Python `dataclasses`
- Include UUID for trade_id
- < 80 lines
- Type hints throughout

Output: Complete file
```

---

## Prompt 4.2 — Filtros de Probabilidad

```markdown
Act as a Quantitative Trading Filter Designer.

Task: Create `execution/filters.py` with probability filtering logic.

---

**Specifications:**

```python
from config.constants import SIGNAL_TYPES
from config.settings import Settings, Thresholds

def classify_probability(
    prob: float, 
    thresholds: Thresholds
) -> SIGNAL_TYPES:
    """
    Classify model probability into action category.
    
    Logic:
    - prob >= confidence_threshold_high → REAL_TRADE
    - learning_min <= prob < learning_max → SHADOW_TRADE  
    - prob < learning_min → IGNORE
    
    Args:
        prob: Model output probability (0-1)
        thresholds: Threshold configuration
    
    Returns:
        SIGNAL_TYPES enum value
    """


def filter_signals(
    model_outputs: Dict[str, float],
    settings: Settings,
    timestamp: datetime
) -> List[TradeSignal]:
    """
    Convert raw model probabilities to actionable signals.
    
    Args:
        model_outputs: Dict with 'rise_fall', 'touch', 'range' probabilities
        settings: Full settings object
        timestamp: Current timestamp
    
    Returns:
        List of TradeSignal objects (may include IGNORE signals)
    
    Flow:
    1. Iterate each contract type
    2. Apply classify_probability
    3. Determine direction (for Rise/Fall)
    4. Create TradeSignal object
    """


def get_actionable_signals(
    signals: List[TradeSignal]
) -> Tuple[List[TradeSignal], List[TradeSignal]]:
    """
    Separate signals into real and shadow trades.
    
    Returns:
        (real_trades, shadow_trades) - IGNORE signals excluded
    """
```

---

**Constraints:**
- Pure functions (no side effects)
- < 100 lines
- Clear docstrings with logic explanation

Output: Complete file
```

---

## Prompt 4.3 — Logger de Shadow Trading

```markdown
Act as a Data Collection Engineer.

Task: Create `execution/shadow_logger.py` for shadow trade persistence.

---

**Specifications:**

```python
from pathlib import Path
import json
import threading
from typing import List, Optional

class ShadowLogger:
    """
    Persistent logger for shadow (paper) trades.
    
    Uses NDJSON (Newline Delimited JSON) for:
    - Append-only efficiency
    - Easy streaming reads
    - Line-by-line processing
    """
    
    def __init__(self, log_path: Path):
        """
        Args:
            log_path: Path to NDJSON log file
        """
        self._log_path = log_path
        self._lock = threading.Lock()  # Thread safety
    
    def log_trade(self, trade: ShadowTrade) -> None:
        """
        Append trade to log file.
        
        Thread-safe append operation.
        """
    
    def load_trades(
        self, 
        since: Optional[datetime] = None
    ) -> List[ShadowTrade]:
        """
        Load trades from log file.
        
        Args:
            since: Optional filter for trades after this timestamp
        """
    
    def update_outcome(
        self, 
        trade_id: str, 
        outcome: bool,
        exit_price: float
    ) -> bool:
        """
        Update specific trade with outcome.
        
        Note: For NDJSON, this requires rewriting the file.
        Consider using a database for frequent updates.
        
        Returns: True if trade found and updated
        """
    
    def export_to_training_data(
        self, 
        output_path: Path,
        only_completed: bool = True
    ) -> int:
        """
        Export shadow trades as labeled training dataset.
        
        Args:
            output_path: Output CSV/parquet path
            only_completed: Only include trades with outcomes
        
        Returns: Number of records exported
        """
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Compute summary statistics of logged trades.
        
        Returns: Dict with win_rate, total_trades, by_contract_type, etc.
        """
```

---

**Constraints:**
- Thread-safe operations
- NDJSON format
- Robust error handling for I/O
- < 120 lines
- Docstrings noting scalability concerns

Output: Complete file
```

---

## Prompt 4.4 — Motor de Decisión

```markdown
Act as a Trading Logic Coordinator.

Task: Create `execution/decision.py` orchestrating the complete decision flow.

---

**Specifications:**

```python
from typing import Dict, List, Any
from datetime import datetime

class DecisionEngine:
    """
    Central coordinator for trading decisions.
    
    Responsibilities:
    1. Process model outputs
    2. Filter probabilities into signals
    3. Log shadow trades
    4. Return actionable real trades
    """
    
    def __init__(
        self, 
        settings: Settings,
        shadow_logger: Optional[ShadowLogger] = None
    ):
        """
        Args:
            settings: Application settings
            shadow_logger: Optional logger for shadow trades
        """
        self.settings = settings
        self.shadow_logger = shadow_logger
        self._stats = {'processed': 0, 'real': 0, 'shadow': 0, 'ignored': 0}
    
    def process_model_output(
        self,
        probs: Dict[str, float],
        timestamp: Optional[datetime] = None,
        market_data: Optional[Dict[str, Any]] = None
    ) -> List[TradeSignal]:
        """
        Main entry point for decision making.
        
        Args:
            probs: Model probability outputs
            timestamp: Optional timestamp (defaults to now)
            market_data: Optional market context for logging
        
        Returns:
            List of REAL_TRADE signals only
        
        Side effects:
            - Shadow trades logged to shadow_logger
            - Internal stats updated
        """
    
    def analyze_opportunity(
        self,
        probs: Dict[str, float],
        timestamp: datetime
    ) -> Tuple[List[TradeSignal], List[TradeSignal], List[TradeSignal]]:
        """
        Detailed analysis returning all signal categories.
        
        Returns:
            (real_trades, shadow_trades, ignored) tuples
        """
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get processing statistics.
        
        Returns: Dict with counts, rates, and shadow trade stats
        """
    
    def get_shadow_performance(self) -> Optional[Dict[str, float]]:
        """
        Calculate shadow trade performance metrics.
        
        Returns: Dict with win_rate, avg_confidence, profit_factor, etc.
                 None if no shadow logger configured
        """
```

---

**Constraints:**
- Clean orchestration (no direct model I/O)
- < 100 lines
- Clear separation of concerns
- Comprehensive docstrings

Output: Complete file
```

---

# 📋 Ventajas de Esta Arquitectura

| Aspecto | Beneficio |
|---------|-----------|
| **Archivos Pequeños** (40-200 líneas) | Más fáciles de generar/depurar por IA |
| **Responsabilidad Única** | Cada módulo tiene un propósito claro |
| **Composabilidad** | Bloques reutilizables reducen duplicación |
| **Testeable** | Funciones puras y clases pequeñas facilitan unit tests |
| **Escalable** | Agregar nuevos experts/heads sin tocar código existente |
| **Colaboración** | Múltiples desarrolladores pueden trabajar en paralelo |
| **Mantenible** | Cambios aislados, bajo riesgo de regresiones |

---

# 📦 Dependencias Requeridas

```txt
# requirements.txt
pydantic>=1.10,<2.0  # or pydantic>=2.0 with BaseSettings from pydantic-settings
torch>=2.0
numpy>=1.24
python-dotenv>=1.0
```

---

# ✅ Checklist de Implementación

- [ ] **Fase 1**: Fundamentos
  - [ ] `config/constants.py`
  - [ ] `utils/device.py`
  - [ ] `utils/seed.py`
  - [ ] `config/settings.py`
  - [ ] `.env`

- [ ] **Fase 2**: Data Pipeline
  - [ ] `data/normalizers.py`
  - [ ] `data/indicators.py`
  - [ ] `data/preprocessor.py`
  - [ ] `data/dataset.py`
  - [ ] `data/loader.py`

- [ ] **Fase 3**: Modelos
  - [ ] `models/attention.py`
  - [ ] `models/blocks.py`
  - [ ] `models/temporal.py`
  - [ ] `models/spatial.py`
  - [ ] `models/volatility.py`
  - [ ] `models/fusion.py`
  - [ ] `models/heads.py`
  - [ ] `models/core.py`

- [ ] **Fase 4**: Ejecución
  - [ ] `execution/signals.py`
  - [ ] `execution/filters.py`
  - [ ] `execution/shadow_logger.py`
  - [ ] `execution/decision.py`

---

> **Nota**: Ejecuta los prompts en orden. Cada prompt asume que los anteriores ya fueron implementados.
