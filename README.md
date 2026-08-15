# Gait Phase Detection

Python application for step detection and gait phase estimation from accelerometer data.

The project processes acceleration measurements to identify individual steps, estimate gait phases and calculate basic gait parameters.

## Features

* Import accelerometer measurement data from CSV files
* Calculate acceleration magnitude from three-axis measurements
* Estimate signal sampling frequency
* Apply Butterworth low-pass filtering
* Detect individual steps using peak detection
* Calculate step count and cadence
* Approximate stance and swing phases of the gait cycle
* Generate processed CSV files and signal visualizations

## Tech Stack

* Python
* NumPy
* Pandas
* SciPy
* Matplotlib

## Running the Application

### 1. Clone the repository

```bash
git clone https://github.com/m-sadlowski/gait-phase-detection.git
cd gait-phase-detection
```

### 2. Create and activate a virtual environment

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Linux / macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the application

Run the project using the provided Python entry point and supply the accelerometer measurement data to be processed.

## Output

The application generates:

* detected step information,
* estimated stance and swing phases,
* gait cadence,
* processed measurement data,
* plots presenting detected steps and gait phases.