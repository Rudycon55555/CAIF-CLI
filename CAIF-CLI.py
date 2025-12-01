# CAIF-CLI.py - Code AI Fast Command Line Interface (Refined with pure-Python AI math)
# Fully dependency-free and portable: uses only Python standard library modules.
# Implements a minimal single-hidden-layer neural network with pure-Python forward/backward passes,
# training loop, dataset parsing for simple numeric CSV/TSV pairs, and an EXPORT MODEL that writes
# a standalone dependency-free .py file containing learned weights and a working predict().

import sys
import os
import socket
import math
import random

# ======================================================================
# UTILITIES
# ======================================================================

def _coerce_value(s):
    """Best-effort type coercion: int, float, bool, or keep as string (quotes removed)."""
    v = s.strip()
    if (v.startswith('"') and v.endswith('"')) or (v.startswith("'") and v.endswith("'")):
        return v[1:-1]
    low = v.lower()
    if low == 'true':
        return True
    if low == 'false':
        return False
    try:
        return int(v)
    except Exception:
        pass
    try:
        return float(v)
    except Exception:
        pass
    return v

def _split_quoted_csv(s):
    """
    Split a CSV-like string that may contain quoted segments.
    Example: '"a, b", c, "d"' -> ['a, b', 'c', 'd']
    """
    items = []
    buf = []
    in_quote = False
    quote_char = None
    for ch in s:
        if in_quote:
            if ch == quote_char:
                in_quote = False
                quote_char = None
            else:
                buf.append(ch)
        else:
            if ch in ('"', "'"):
                in_quote = True
                quote_char = ch
            elif ch == ',':
                item = ''.join(buf).strip()
                if item:
                    items.append(item)
                buf = []
            else:
                buf.append(ch)
    last = ''.join(buf).strip()
    if last:
        items.append(last)
    return items

# ======================================================================
# SIMPLE DATA LOADER (Parses numeric CSV/TSV pairs into dataset)
# ======================================================================

def _parse_numeric_pairs_from_text(text):
    """
    Parse lines of the form:
      x1,x2,...,xn \t y1,y2,...,ym
    or
      x1,x2,...,xn , y1,y2,...,ym
    Returns list of (input_list, target_list).
    Non-numeric lines are ignored.
    """
    dataset = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith('#'):
            continue
        # Try tab-separated input/target first
        if '\t' in line:
            left, right = line.split('\t', 1)
        elif '|' in line:
            left, right = line.split('|', 1)
        elif ',' in line and ';' not in line:
            # Heuristic: if there are two groups separated by ' , ' or ' ,'
            # Try to split by '  ' (double space) or by last comma group
            parts = line.split(',')
            # If even number of parts, assume half inputs half outputs is ambiguous.
            # Simpler heuristic: if there's exactly 2 columns separated by comma-space
            if ', ' in line:
                # split on ' , ' or ', ' into two groups at the first occurrence of ' , '
                # fallback: split into two halves
                # We'll attempt to split at the middle comma if there are exactly 2 groups
                # Try splitting by ' , ' first
                if ' , ' in line:
                    left, right = line.split(' , ', 1)
                else:
                    # fallback: split into two halves
                    mid = len(parts) // 2
                    left = ','.join(parts[:mid])
                    right = ','.join(parts[mid:])
            else:
                # fallback: treat entire line as input and ignore
                continue
        else:
            # Not a numeric pair line
            continue

        # Convert left and right into numeric lists
        def to_num_list(s):
            tokens = [t.strip() for t in s.replace(';', ',').split(',') if t.strip()]
            nums = []
            for tok in tokens:
                try:
                    if '.' in tok or 'e' in tok.lower():
                        nums.append(float(tok))
                    else:
                        nums.append(int(tok))
                except Exception:
                    # Non-numeric token -> abort
                    return None
            return [float(x) for x in nums]

        xin = to_num_list(left)
        yout = to_num_list(right)
        if xin is None or yout is None:
            continue
        dataset.append((xin, yout))
    return dataset

def _load_data(filepaths, config):
    """Handles IMPORT DATA from local files or URLs. Stores parsed numeric pairs in config['dataset']."""
    import urllib.request  # standard library
    print(f"[DATA] Reading data from sources: {filepaths}")

    config.setdefault('data_files', [])
    config.setdefault('dataset', [])

    for source in filepaths:
        try:
            if source.startswith('http://') or source.startswith('https://'):
                with urllib.request.urlopen(source) as response:
                    content = response.read().decode('utf-8', errors='replace')
                    print(f"   -> Fetched {len(content)} bytes from URL.")
            else:
                with open(source, 'r', encoding='utf-8') as f:
                    content = f.read()
                    print(f"   -> Read {len(content)} characters from local file.")
            parsed = _parse_numeric_pairs_from_text(content)
            if parsed:
                config['dataset'].extend(parsed)
                print(f"   -> Parsed {len(parsed)} numeric pairs from '{source}'.")
            else:
                # If no numeric pairs, store raw content length as conceptual data
                config['dataset_size'] = config.get('dataset_size', 0) + len(content)
                print(f"   -> No numeric pairs found in '{source}', stored conceptual size.")
            config['data_files'].append(source)
        except Exception as e:
            print(f"[ERROR] Data load failed for '{source}': {e}")

    print(f"[DATA] Total dataset entries: {len(config.get('dataset', []))}")
    return True

# ======================================================================
# PURE-PYTHON NEURAL NETWORK (Single hidden layer, MSE loss, SGD)
# ======================================================================

def _init_weights(input_dim, hidden_dim, output_dim, seed=None):
    """Initialize weights and biases with small random values (pure Python)."""
    if seed is not None:
        random.seed(seed)
    # Heuristic scale
    def rand_matrix(rows, cols, scale=0.1):
        return [[(random.random() * 2 - 1) * scale for _ in range(cols)] for _ in range(rows)]
    W1 = rand_matrix(hidden_dim, input_dim)   # hidden_dim x input_dim
    b1 = [0.0 for _ in range(hidden_dim)]
    W2 = rand_matrix(output_dim, hidden_dim)  # output_dim x hidden_dim
    b2 = [0.0 for _ in range(output_dim)]
    return {'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}

def _vec_add(a, b):
    return [x + y for x, y in zip(a, b)]

def _vec_sub(a, b):
    return [x - y for x, y in zip(a, b)]

def _vec_mul_scalar(a, s):
    return [x * s for x in a]

def _dot_row_vec(row, vec):
    s = 0.0
    for i in range(len(row)):
        s += row[i] * vec[i]
    return s

def _apply_tanh(v):
    return [math.tanh(x) for x in v]

def _apply_tanh_derivative(v):
    # v is tanh(x) values; derivative = 1 - tanh^2(x)
    return [1.0 - (x * x) for x in v]

def _forward(weights, x):
    """Forward pass: returns hidden_activation (tanh) and output (linear)."""
    W1, b1, W2, b2 = weights['W1'], weights['b1'], weights['W2'], weights['b2']
    # hidden = tanh(W1 * x + b1)
    hidden_raw = []
    for i in range(len(W1)):
        hidden_raw.append(_dot_row_vec(W1[i], x) + b1[i])
    hidden = _apply_tanh(hidden_raw)
    # output = W2 * hidden + b2 (linear output)
    out = []
    for i in range(len(W2)):
        out.append(_dot_row_vec(W2[i], hidden) + b2[i])
    return hidden, out

def _mse_loss(pred, target):
    s = 0.0
    for p, t in zip(pred, target):
        d = p - t
        s += d * d
    return s / len(pred)

def _backward(weights, x, hidden, pred, target, lr):
    """Backward pass: compute gradients and update weights in-place using SGD."""
    W1, b1, W2, b2 = weights['W1'], weights['b1'], weights['W2'], weights['b2']
    # Output layer gradient (dL/dout) for MSE: 2*(pred - target)/n
    n_out = len(pred)
    d_out = [2.0 * (pred[i] - target[i]) / n_out for i in range(n_out)]
    # Gradients for W2 and b2
    # dW2[i][j] = d_out[i] * hidden[j]
    for i in range(len(W2)):
        for j in range(len(W2[0])):
            grad = d_out[i] * hidden[j]
            W2[i][j] -= lr * grad
        b2[i] -= lr * d_out[i]
    # Backprop into hidden: dh = sum_i (W2_i * d_out_i)
    dh = [0.0 for _ in range(len(hidden))]
    for j in range(len(hidden)):
        s = 0.0
        for i in range(len(W2)):
            s += W2[i][j] * d_out[i]
        dh[j] = s
    # Multiply by tanh derivative
    tanh_deriv = _apply_tanh_derivative(hidden)
    dh = [dh[j] * tanh_deriv[j] for j in range(len(dh))]
    # Gradients for W1 and b1
    for i in range(len(W1)):
        for j in range(len(W1[0])):
            grad = dh[i] * x[j]
            W1[i][j] -= lr * grad
        b1[i] -= lr * dh[i]
    # weights updated in-place

# ======================================================================
# HIGH-LEVEL MODEL LIFECYCLE FUNCTIONS
# ======================================================================

def _initialize_model(model_type, parameters):
    """Initialize model metadata and weights based on parameters."""
    print(f"[MODEL] Initializing {model_type} structure.")
    input_dim = int(parameters.get('input_dim', 1))
    hidden_neurons = int(parameters.get('hidden_neurons', 16))
    output_dim = int(parameters.get('output_dim', 1))
    seed = parameters.get('seed', None)
    weights = _init_weights(input_dim, hidden_neurons, output_dim, seed=seed)
    model = {
        'type': model_type,
        'is_trained': False,
        'context_map': {},
        'weights': weights,
        'input_dim': input_dim,
        'hidden_neurons': hidden_neurons,
        'output_dim': output_dim
    }
    print(f"   -> input_dim={input_dim}, hidden_neurons={hidden_neurons}, output_dim={output_dim}")
    return model

def _run_training(model, epochs, mode="supervised", config=None):
    """Train the model using dataset in config['dataset'] with pure-Python SGD."""
    if not model:
        raise ValueError("Model must be initialized before training.")
    if config is None:
        config = {}
    dataset = config.get('dataset', [])
    if not dataset:
        print("[TRAIN] No numeric dataset available; running dummy epochs to simulate training.")
        for i in range(1, int(epochs) + 1):
            print(f"   -> Epoch {i}/{epochs} complete (no data).")
        model['is_trained'] = True
        return model

    lr = float(config.get('parameters', {}).get('learning_rate', 0.01))
    batch_size = int(config.get('parameters', {}).get('batch_size', 1))
    print(f"[TRAIN] Starting {mode} training for {epochs} epochs on {len(dataset)} samples (lr={lr}, batch={batch_size})")
    for epoch in range(1, int(epochs) + 1):
        # Simple SGD: shuffle dataset each epoch
        random.shuffle(dataset)
        total_loss = 0.0
        for idx in range(0, len(dataset), batch_size):
            batch = dataset[idx: idx + batch_size]
            # For each sample in batch, do forward/backward
            for x_raw, y_raw in batch:
                # Ensure input/output dims match model
                x = list(x_raw)
                y = list(y_raw)
                # Pad or truncate input/output to model dims
                if len(x) < model['input_dim']:
                    x = x + [0.0] * (model['input_dim'] - len(x))
                else:
                    x = x[:model['input_dim']]
                if len(y) < model['output_dim']:
                    y = y + [0.0] * (model['output_dim'] - len(y))
                else:
                    y = y[:model['output_dim']]
                hidden, pred = _forward(model['weights'], x)
                loss = _mse_loss(pred, y)
                total_loss += loss
                _backward(model['weights'], x, hidden, pred, y, lr)
        avg_loss = total_loss / max(1, len(dataset))
        print(f"   -> Epoch {epoch}/{epochs} complete. Avg Loss: {avg_loss:.6f}")
    model['is_trained'] = True
    print("[TRAIN] Training complete. Weights updated.")
    return model

def _set_connection(model, target_type, address, context):
    """Store connection details in model context_map."""
    if not model:
        raise ValueError("Model must be initialized before setting connections.")
    model['context_map'][context] = {'type': target_type, 'address': address}
    print(f"[CONNECT] Stored {target_type} connection to '{address}' as '{context}'")
    return model

# ======================================================================
# EXPORT: write a standalone .py file containing learned weights and predict()
# ======================================================================

def _export_model(model, name, parameters):
    """
    Export the trained model into a single dependency-free Python file.
    The exported file contains the learned weights and a working predict() that
    performs the same forward math as the training runtime.
    """
    output_filename = f"{name}.py"
    weights = model.get('weights', {})
    context_map = model.get('context_map', {})
    input_dim = model.get('input_dim', 1)
    hidden_neurons = model.get('hidden_neurons', 1)
    output_dim = model.get('output_dim', 1)

    # Serialize lists in a compact, readable way
    def repr_list(obj):
        return repr(obj)

    with open(output_filename, 'w', encoding='utf-8') as out:
        out.write("# Exported CAIF model (dependency-free)\n")
        out.write("# This file contains learned weights and a predict(prompt_vector) function.\n\n")
        out.write("import math\n")
        out.write("import socket\n\n")
        out.write(f"INPUT_DIM = {input_dim}\n")
        out.write(f"HIDDEN_NEURONS = {hidden_neurons}\n")
        out.write(f"OUTPUT_DIM = {output_dim}\n\n")
        out.write(f"MODEL_CONTEXT = {repr(context_map)}\n\n")
        out.write("# Learned weights and biases\n")
        out.write(f"W1 = {repr_list(weights.get('W1', []))}\n")
        out.write(f"b1 = {repr_list(weights.get('b1', []))}\n")
        out.write(f"W2 = {repr_list(weights.get('W2', []))}\n")
        out.write(f"b2 = {repr_list(weights.get('b2', []))}\n\n")

        out.write("def _dot_row_vec(row, vec):\n")
        out.write("    s = 0.0\n")
        out.write("    for i in range(len(row)):\n")
        out.write("        s += row[i] * vec[i]\n")
        out.write("    return s\n\n")

        out.write("def _apply_tanh(v):\n")
        out.write("    return [math.tanh(x) for x in v]\n\n")

        out.write("def _send_xp_command(command_data, target_address):\n")
        out.write("    try:\n")
        out.write("        host, port_str = target_address.split(':')\n")
        out.write("        port = int(port_str)\n")
        out.write("    except Exception:\n")
        out.write("        print('ERROR: Invalid XP address format (must be host:port).')\n")
        out.write("        return False\n")
        out.write("    try:\n")
        out.write("        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)\n")
        out.write("        s.connect((host, port))\n")
        out.write("        s.sendall((str(command_data)).encode('utf-8') + b'\\n')\n")
        out.write("        s.close()\n")
        out.write("        return True\n")
        out.write("    except Exception as e:\n")
        out.write("        print(f'XP Connection Failed: {e}')\n")
        out.write("        return False\n\n")

        out.write("def _run_tc_tool(tool_name, arguments):\n")
        out.write("    # USER HOOK: replace with local tool logic if desired\n")
        out.write("    print(f'Tool Call Requested: {tool_name}({arguments})')\n")
        out.write("    return f\"Tool '{tool_name}' executed placeholder logic with args: {arguments}\"\n\n")

        out.write("def predict(input_vector):\n")
        out.write("    # input_vector: list of floats of length INPUT_DIM (will be padded/truncated)\n")
        out.write("    x = list(input_vector)\n")
        out.write("    if len(x) < INPUT_DIM:\n")
        out.write("        x = x + [0.0] * (INPUT_DIM - len(x))\n")
        out.write("    else:\n")
        out.write("        x = x[:INPUT_DIM]\n\n")
        out.write("    # Hidden layer\n")
        out.write("    hidden_raw = []\n")
        out.write("    for i in range(len(W1)):\n")
        out.write("        hidden_raw.append(_dot_row_vec(W1[i], x) + b1[i])\n")
        out.write("    hidden = _apply_tanh(hidden_raw)\n\n")
        out.write("    # Output layer (linear)\n")
        out.write("    out = []\n")
        out.write("    for i in range(len(W2)):\n")
        out.write("        out.append(_dot_row_vec(W2[i], hidden) + b2[i])\n\n")
        out.write("    # Simple action decoding heuristic: choose action based on output magnitudes\n")
        out.write("    a = out[0] if len(out) > 0 else 0.0\n")
        out.write("    b = out[1] if len(out) > 1 else 0.0\n")
        out.write("    if a > b and a > 1.0:\n")
        out.write("        # XP action: send to first XP context if available\n")
        out.write("        xp_ctx = None\n")
        out.write("        for ctx, info in MODEL_CONTEXT.items():\n")
        out.write("            if (info or {}).get('type') == 'XP':\n")
        out.write("                xp_ctx = info\n")
        out.write("                break\n")
        out.write("        if xp_ctx:\n")
        out.write("            _send_xp_command({'command': 'AUTO', 'payload': out}, xp_ctx.get('address'))\n")
        out.write("            return f'XP Command Sent to {xp_ctx.get(\"address\")}: {out}'\n")
        out.write("        return 'No XP context configured.'\n")
        out.write("    if b > a and b > 1.0:\n")
        out.write("        tc_ctx = None\n")
        out.write("        for ctx, info in MODEL_CONTEXT.items():\n")
        out.write("            if (info or {}).get('type') == 'TC':\n")
        out.write("                tc_ctx = info\n")
        out.write("                break\n")
        out.write("        tool_name = (tc_ctx or {}).get('address', 'default_tool')\n")
        out.write("        return _run_tc_tool(tool_name, {'output': out})\n")
        out.write("    return out\n\n")

        out.write("if __name__ == '__main__':\n")
        out.write("    print('\\nRunning exported model...\\n')\n")
        out.write("    # Example: predict with zero vector\n")
        out.write("    sample = [0.0] * INPUT_DIM\n")
        out.write("    print('Prediction:', predict(sample))\n")

    print(f"[SUCCESS] Exported model to {output_filename}")
    return True

# ======================================================================
# COMMAND LINE PARSER
# ======================================================================

def execute_caif_file(filepath):
    """Parses and executes CAIF commands line by line."""
    if not os.path.exists(filepath):
        print(f"[ERROR] CAIF file not found at '{filepath}'")
        return

    print("=========================================")
    print(f"  CAIF Executor started for: {filepath}")
    print("=========================================")

    config = {'parameters': {}, 'data_files': [], 'dataset': []}
    current_model = None

    with open(filepath, 'r', encoding='utf-8') as f:
        for line_num, raw_line in enumerate(f, 1):
            line = raw_line.strip()
            if not line or line.startswith('#'):
                continue

            try:
                if line.startswith('SET PARAMETER'):
                    tail = line[len('SET PARAMETER'):].strip()
                    if '=' not in tail:
                        print(f"[ERROR] Malformed SET PARAMETER on line {line_num}: missing '='")
                        continue
                    key, value = tail.split('=', 1)
                    key = key.strip()
                    value = _coerce_value(value)
                    config['parameters'][key] = value
                    print(f"[PARAM] Set {key} = {value}")

                elif line.startswith('IMPORT DATA'):
                    tail = line[len('IMPORT DATA'):].strip()
                    paths = _split_quoted_csv(tail)
                    if not paths:
                        print(f"[ERROR] IMPORT DATA requires at least one source (line {line_num}).")
                        continue
                    _load_data(paths, config)

                elif line.startswith('START MODEL'):
                    tail = line[len('START MODEL'):].strip()
                    model_type = _coerce_value(tail)
                    # Merge parameters into config for training use
                    merged = {'parameters': config.get('parameters', {})}
                    merged.update(config)
                    current_model = _initialize_model(model_type, config['parameters'])

                elif line.startswith('TRAIN FOR'):
                    tail = line[len('TRAIN FOR'):].strip()
                    parts = tail.split()
                    if not parts or not parts[0].isdigit():
                        print(f"[ERROR] TRAIN FOR requires integer epoch count (line {line_num}).")
                        continue
                    epochs = int(parts[0])
                    current_model = _run_training(current_model, epochs, mode="supervised", config={'dataset': config.get('dataset', []), 'parameters': config.get('parameters', {})})

                elif line.startswith('START TRAINING'):
                    tail = line[len('START TRAINING'):].strip()
                    mode = "unsupervised"
                    epochs = 5
                    tokens = tail.split()
                    i = 0
                    while i < len(tokens):
                        tok = tokens[i].lower()
                        if tok == 'for' and i + 1 < len(tokens):
                            mode = tokens[i + 1]
                            i += 2
                        elif tok == 'epochs' and i + 1 < len(tokens):
                            try:
                                epochs = int(tokens[i + 1])
                            except Exception:
                                print(f"[WARN] Invalid epochs value on line {line_num}, defaulting to {epochs}.")
                            i += 2
                        else:
                            i += 1
                    current_model = _run_training(current_model, epochs=epochs, mode=mode, config={'dataset': config.get('dataset', []), 'parameters': config.get('parameters', {})})

                elif line.startswith('CONNECT TO'):
                    tail = line[len('CONNECT TO'):].strip()
                    parts = tail.split(None, 2)
                    if len(parts) < 2:
                        print(f"[ERROR] Malformed CONNECT TO on line {line_num}.")
                        continue
                    target_type = parts[0].strip()
                    rest = parts[1] if len(parts) >= 2 else ''
                    remainder = parts[2] if len(parts) == 3 else ''
                    addr = None
                    ctx = 'default'
                    combined = (rest + ' ' + remainder).strip()
                    q_start = None
                    q_end = None
                    q_ch = None
                    for idx, ch in enumerate(combined):
                        if ch in ('"', "'"):
                            q_start = idx
                            q_ch = ch
                            break
                    if q_start is not None:
                        for j in range(q_start + 1, len(combined)):
                            if combined[j] == q_ch:
                                q_end = j
                                break
                        if q_end is not None:
                            addr = combined[q_start + 1:q_end]
                            after = combined[q_end + 1:].strip()
                            if after.lower().startswith('as '):
                                ctx = after[3:].strip()
                                ctx = _coerce_value(ctx)
                        else:
                            addr = combined
                    if addr is None:
                        addr = rest.strip()
                    current_model = _set_connection(current_model, target_type, addr, ctx)

                elif line.startswith('EXPORT MODEL'):
                    if not current_model or not current_model.get('is_trained'):
                        print(f"[ERROR] Cannot EXPORT MODEL before training on line {line_num}.")
                        continue
                    tail = line[len('EXPORT MODEL'):].strip()
                    if tail.lower().startswith('as'):
                        model_name = _coerce_value(tail[2:].strip())
                    else:
                        model_name = _coerce_value(tail)
                    if not model_name:
                        print(f"[ERROR] EXPORT MODEL requires a name (line {line_num}).")
                        continue
                    _export_model(current_model, model_name, config.get('parameters', {}))

                else:
                    print(f"[ERROR] Unknown command on line {line_num}: {line}")

            except Exception as e:
                print(f"[FATAL] CRASH on line {line_num} ({line}): {e}")

    print("=========================================")
    print("  CAIF Execution Finished.")
    print("=========================================")

# ======================================================================
# ENTRY POINT
# ======================================================================

if __name__ == "__main__":
    if len(sys.argv) > 1:
        execute_caif_file(sys.argv[1])
    else:
        print("Usage: python CAIF-CLI.py <path_to_caif_file>")
        print("\nExample CAIF script contents:")
        print('''# Example.caif
SET PARAMETER input_dim = 2
SET PARAMETER output_dim = 1
SET PARAMETER hidden_neurons = 8
SET PARAMETER learning_rate = 0.05
IMPORT DATA "train_pairs.txt"
START MODEL "regressor"
TRAIN FOR 20
CONNECT TO XP "127.0.0.1:9000" as robot_arm
EXPORT MODEL as "MyTrainedModel"
''')
