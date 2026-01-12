# Migration Guide: V1 → V2

**From**: Dynamic allocation, Ring buffer  
**To**: Pre-allocated, Lock-free triple buffer, RAM-optimized

---

## 📋 Quick Comparison

| Feature | V1 | V2 |
|---------|----|----|
| **Memory Model** | Dynamic allocation | Pre-allocated (2+8+4=14GB) |
| **Audio Buffer** | Ring buffer (locks) | Triple buffer (lock-free) |
| **Model Loading** | On-demand | Hot cache (8GB) |
| **Feature Extraction** | Every time | Cached (4GB) |
| **Latency** | ~18ms | ~12ms (target) |
| **Stability** | Good | Excellent |
| **Failsafe** | Manual | Automatic |
| **Integrity Checks** | None | Every 60s |
| **RAM Required** | 8GB | 24GB (recommended) |

---

## 🔄 Code Changes

### **1. Buffer Management**

**V1 (Ring Buffer)**:
```python
from app.audio.buffer_ring import RingBuffer

ring_buffer = RingBuffer(capacity_samples=480000, channels=1)
ring_buffer.write(audio_data)
audio_chunk = ring_buffer.read(4096)
```

**V2 (Triple Buffer + Pool)**:
```python
from app.audio.buffer_pool import BufferPool
from app.audio.triple_buffer import TripleBuffer

# Pre-allocate pool
buffer_pool = BufferPool(slot_size_samples=480000, num_slots=100)

# Lock-free communication
triple_buffer = TripleBuffer(buffer_size=48000, channels=1)
triple_buffer.write(audio_data)
audio_chunk = triple_buffer.read()  # Returns None if not ready
```

**Migration**: Replace all `RingBuffer` with `TripleBuffer` + `BufferPool`

---

### **2. Audio Stream**

**V1 (AudioStream)**:
```python
from app.audio.audio_stream import AudioStream

stream = AudioStream(
    input_device_index=0,
    output_device_index=1,
    sample_rate=48000,
    buffer_size=256
)
stream.start()
```

**V2 (AudioStreamV2)**:
```python
from app.audio.audio_stream_v2 import AudioStreamV2

# Create buffer pool first
buffer_pool = BufferPool(...)

stream = AudioStreamV2(
    input_device_index=0,
    output_device_index=1,
    sample_rate=48000,
    buffer_size=256,
    buffer_pool=buffer_pool  # NEW: Pass buffer pool
)

# Start with validation
if not stream.start():
    logger.error("Failed to start")
```

**Migration**: 
1. Create `BufferPool` before stream
2. Pass to `AudioStreamV2`
3. Check return value of `start()`

---

### **3. Model Loading**

**V1 (Direct Load)**:
```python
from app.core.model_loader import ModelLoader

loader = ModelLoader(device="cuda")
model_data = loader.load_model(pth_path, index_path)
```

**V2 (Cached Loading)**:
```python
from app.core.model_cache import ModelCache
from app.core.model_loader import ModelLoader

# Create cache
model_cache = ModelCache(max_cache_size_gb=8.0, device="cuda")

# Try cache first
model_data = model_cache.get(model_id)

if model_data is None:
    # Load and cache
    loader = ModelLoader(device="cuda")
    model_data = loader.load_model(pth_path, index_path)
    model_cache.put(model_id, model_data, warmup=True)
```

**Migration**:
1. Create `ModelCache` at startup
2. Always check cache before loading
3. Cache after loading

---

### **4. Main Application**

**V1 (main.py)**:
```python
from app.main import VoiceForgeApp

app = VoiceForgeApp()
app.initialize()
app.start_golden_path()
```

**V2 (main_v2.py)**:
```python
from app.main_v2 import VoiceForgeAppV2

app = VoiceForgeAppV2()

# NEW: Returns bool, check before starting
if not app.initialize():
    logger.error("Init failed")
    sys.exit(1)

if not app.start_golden_path():
    logger.error("Start failed")
    sys.exit(1)
```

**Migration**:
1. Import from `main_v2`
2. Check return values
3. Handle failures gracefully

---

## 🔧 Configuration Changes

### **V1 (Hardcoded)**:
```python
# In code
buffer_size = 256
sample_rate = 48000
```

### **V2 (config.yaml)**:
```yaml
audio:
  sample_rate: 48000
  buffer_size: 256
  
resources:
  allocation:
    audio_buffers_gb: 2
    model_cache_gb: 8
    feature_cache_gb: 4
```

**Migration**: Move all constants to `config.yaml`

---

## 📦 New Dependencies

Add to `requirements.txt`:

```txt
# V2 specific (if not already present)
pyyaml==6.0.1          # For config.yaml
```

---

## 🗂️ New Files to Create

```
app/
├── audio/
│   ├── buffer_pool.py        # NEW
│   ├── triple_buffer.py      # NEW
│   └── audio_stream_v2.py    # REPLACES audio_stream.py
├── core/
│   ├── model_cache.py        # NEW
│   └── feature_cache.py      # NEW
└── main_v2.py                # REPLACES main.py

config.yaml                    # NEW
```

---

## 🚀 Migration Steps

### **Step 1: Backup V1**

```bash
cd VoiceForge-Nextgen
git checkout -b v1-backup
git commit -am "Backup V1 before migration"
```

### **Step 2: Copy New Files**

Copy from artifacts:
- `app/audio/buffer_pool.py`
- `app/audio/triple_buffer.py`
- `app/audio/audio_stream_v2.py`
- `app/core/model_cache.py`
- `app/core/feature_cache.py`
- `app/main_v2.py`
- `config.yaml`

### **Step 3: Update Imports**

Old files using V1 components:

```python
# OLD
from app.audio.buffer_ring import RingBuffer
from app.audio.audio_stream import AudioStream

# NEW
from app.audio.triple_buffer import TripleBuffer
from app.audio.buffer_pool import BufferPool
from app.audio.audio_stream_v2 import AudioStreamV2
```

### **Step 4: Test V2 Separately**

```bash
# Run V2 (doesn't affect V1)
python app/main_v2.py
```

Keep V1 intact:
```bash
# V1 still works
python app/main.py
```

### **Step 5: Validate V2**

Run diagnostic:
```bash
python diagnostic_tool.py
```

Check:
- ✅ Buffer pool pre-allocation successful
- ✅ Integrity check passes
- ✅ Golden Path works

### **Step 6: Switch Default**

If V2 stable, rename:
```bash
mv app/main.py app/main_v1_backup.py
mv app/main_v2.py app/main.py
```

---

## ⚠️ Breaking Changes

### **1. Start() Return Value**

**V1**: `void` (always succeeds or crashes)
```python
stream.start()  # No return value
```

**V2**: `bool` (returns False on failure)
```python
if not stream.start():
    handle_error()
```

**Fix**: Check return values

---

### **2. Buffer Pool Required**

**V1**: Stream creates buffers internally
```python
stream = AudioStream(...)  # Buffers auto-created
```

**V2**: Must provide buffer pool
```python
pool = BufferPool(...)
stream = AudioStreamV2(..., buffer_pool=pool)
```

**Fix**: Create pool before stream

---

### **3. Callback Signature**

**V1**: Optional callback
```python
def process(audio: np.ndarray) -> np.ndarray:
    return processed_audio

stream.set_processing_callback(process)
```

**V2**: Same, but validated
```python
# V2 validates callback return
def process(audio: np.ndarray) -> np.ndarray:
    result = do_processing(audio)
    
    # V2 will validate:
    # - result is not None
    # - result.shape == audio.shape
    # - result is finite
    
    return result
```

**Fix**: Ensure callback always returns valid data

---

### **4. Configuration File**

**V1**: Hardcoded constants
```python
SAMPLE_RATE = 48000
BUFFER_SIZE = 256
```

**V2**: Must have `config.yaml`
```yaml
audio:
  sample_rate: 48000
  buffer_size: 256
```

**Fix**: Create `config.yaml` from template

---

## 🐛 Common Migration Issues

### **Issue 1: "Buffer pool exhausted"**

**Cause**: V2 pre-allocates, finite slots

**V1 Behavior**: Dynamic allocation (unlimited)

**V2 Fix**:
```yaml
# Increase pool size
buffer_pool:
  num_slots: 200  # Was 100
```

---

### **Issue 2: "Integrity check failed"**

**Cause**: Buffer pool corruption (shouldn't happen)

**V1**: No integrity checks

**V2 Fix**: 
1. Check logs for details
2. Snapshot captured automatically
3. Report as bug

---

### **Issue 3: High memory usage**

**Cause**: V2 pre-allocates 14GB

**V1**: Only uses what's needed

**V2 Fix**: Reduce cache sizes:
```yaml
resources:
  allocation:
    model_cache_gb: 4    # Was 8
    feature_cache_gb: 2  # Was 4
```

---

### **Issue 4: "State transition invalid"**

**Cause**: V2 has strict state machine

**V1**: Loose state management

**V2 Fix**: Follow valid transitions:
```
STOPPED → STARTING → RUNNING → STOPPING → STOPPED
```

Don't skip states.

---

## 📊 Performance Expectations

After migration:

| Metric | V1 | V2 (Expected) |
|--------|----|----|
| Startup time | ~2s | ~5s (pre-allocation) |
| Latency | 18ms | 12-15ms |
| Model switch | 500ms | 15ms (cached) |
| Memory usage | 4-8GB | 14-18GB |
| Stability | 99% | 99.9% |

---

## ✅ Post-Migration Checklist

- [ ] All new files copied
- [ ] `config.yaml` created
- [ ] Imports updated
- [ ] Return values checked
- [ ] Diagnostic passes
- [ ] Golden Path works
- [ ] Integrity checks pass
- [ ] Memory usage acceptable
- [ ] Latency improved
- [ ] No crashes for 1 hour

---

## 🔙 Rollback Plan

If V2 fails:

```bash
# Quick rollback
git checkout v1-backup

# Or manually
mv app/main.py app/main_v2.py
mv app/main_v1_backup.py app/main.py

python app/main.py  # Back to V1
```

---

## 🎯 Recommended Migration Path

### **Conservative** (Recommended):
1. Week 1: Test V2 in isolation
2. Week 2: Validate stability
3. Week 3: Switch default to V2
4. Week 4: Remove V1 code

### **Aggressive**:
1. Day 1: Copy files
2. Day 2: Test
3. Day 3: Switch

### **Hybrid**:
Keep both versions:
```bash
python app/main.py     # V1
python app/main_v2.py  # V2
```

Choose per session.

---

## 📞 Support

Issues during migration:
1. Check logs: `logs/debug.log`
2. Check snapshots: `logs/snapshots/`
3. Compare V1 vs V2 behavior
4. Report with diagnostic output

---

**Migration complete when**:
- ✅ V2 runs without errors
- ✅ Latency < V1
- ✅ No failsafe activations
- ✅ Stable for 24h continuous

**You can then safely delete V1 code.**