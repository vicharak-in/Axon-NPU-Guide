#!/bin/bash
set -e  # Exit on error
# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration - Modify these as needed

# Model configuration
MODEL_NAME="yolo11n"
MODEL_URL=""  # Leave empty to use local file, or provide download URL

# ONNX export settings
OPSET_VERSION=19
EXPORT_IMGSZ=1280  # Export at 1280x1280 to avoid graph generation issues

# RKNN conversion settings
GRAPH_SIZE="1280,1280" 
PLATFORM="rk3588"  # rk3588, rk3576, etc.
ENABLE_QUANTIZATION=false
DATASET_FILE=""  # Path to dataset.txt for quantization

# Python version for virtual environments
PYTHON_CMD="python3"

# Helper Functions
print_header() {
    echo -e "${BLUE}============================================================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}============================================================================${NC}"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ Error: $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ Warning: $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ $1${NC}"
}

check_command() {
    if ! command -v $1 &> /dev/null; then
        print_error "$1 is not installed or not in PATH"
        exit 1
    fi
}


# Parse Command Line Arguments

show_usage() {
    cat << EOF
Usage: $0 [OPTIONS]

Convert YOLO11n to RKNN format (static shapes, no dynamic conversion).

Options:
    -m, --model NAME            Model name (default: yolo11n)
    -s, --size H,W              Graph size (default: 1280,1280)
    -p, --platform PLATFORM     Target platform (default: rk3588)
    -q, --quantize              Enable INT8 quantization
    -d, --dataset FILE          Dataset file for quantization
    --opset VERSION             ONNX opset version (default: 19)
    --imgsz SIZE                Export image size (default: 1280)
    --url URL                   Download model from URL
    -h, --help                  Show this help message

Examples:
    # Basic conversion at 1280x1280
    $0 -m yolo11n -s "1280,1280"
    
    # Convert with quantization
    $0 -m yolo11n -s "1280,1280" -q -d dataset.txt
    
    # Convert at 640x640 (if needed, though 1280 is recommended)
    $0 -m yolo11n -s "640,640" --imgsz 640

EOF
    exit 0
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -m|--model)
            MODEL_NAME="$2"
            shift 2
            ;;
        -s|--size)
            GRAPH_SIZE="$2"
            shift 2
            ;;
        -p|--platform)
            PLATFORM="$2"
            shift 2
            ;;
        -q|--quantize)
            ENABLE_QUANTIZATION=true
            shift
            ;;
        -d|--dataset)
            DATASET_FILE="$2"
            shift 2
            ;;
        --opset)
            OPSET_VERSION="$2"
            shift 2
            ;;
        --imgsz)
            EXPORT_IMGSZ="$2"
            shift 2
            ;;
        --url)
            MODEL_URL="$2"
            shift 2
            ;;
        -h|--help)
            show_usage
            ;;
        *)
            print_error "Unknown option: $1"
            show_usage
            ;;
    esac
done


# Pre-flight Checks

print_header "Pre-flight Checks"

# Check Python
check_command $PYTHON_CMD
PYTHON_VERSION=$($PYTHON_CMD --version 2>&1 | awk '{print $2}' | cut -d. -f1,2)
print_success "Python $PYTHON_VERSION detected"

# Check if running in project directory
if [ ! -d "utils" ]; then
    print_error "Please run this script from the project root directory, or create the utils/ folder and place the required scripts there."
    exit 1
fi
print_success "Running from project root"

# Check for required scripts
REQUIRED_SCRIPTS=("utils/removenodes11n.py" "utils/onnx2rknn.py")
for script in "${REQUIRED_SCRIPTS[@]}"; do
    if [ ! -f "$script" ]; then
        print_error "Required script not found: $script"
        exit 1
    fi
done
print_success "All required scripts found"

# Validate quantization settings
if [ "$ENABLE_QUANTIZATION" = true ] && [ -z "$DATASET_FILE" ]; then
    print_error "Quantization enabled but no dataset file specified. Use -d option."
    exit 1
fi

if [ "$ENABLE_QUANTIZATION" = true ] && [ ! -f "$DATASET_FILE" ]; then
    print_error "Dataset file not found: $DATASET_FILE"
    exit 1
fi

echo ""

# Setup Virtual Environment: venv-onnx

print_header "Step 1: Setup ONNX Virtual Environment"

if [ ! -d "venv-onnx" ]; then
    print_info "Creating venv-onnx..."
    $PYTHON_CMD -m venv venv-onnx
    print_success "Created venv-onnx"
    
    source venv-onnx/bin/activate
    print_info "Installing ONNX environment packages..."
    pip install --upgrade pip
    pip install ultralytics
    pip install onnx
    pip install onnx-graphsurgeon==0.5.2
    pip install numpy
    deactivate
    print_success "ONNX environment setup complete"
else
    print_success "venv-onnx already exists"
fi

echo ""

# Setup Virtual Environment: venv-rknn

print_header "Step 2: Setup RKNN Virtual Environment"

# Create venv if it doesn't exist
if [ ! -d "venv-rknn" ]; then
    print_info "Creating venv-rknn..."
    $PYTHON_CMD -m venv venv-rknn
    print_success "Created venv-rknn"
fi

# Check if RKNN toolkit is installed
source venv-rknn/bin/activate
RKNN_INSTALLED=$($PYTHON_CMD -c "try:
    from rknn.api import RKNN
    print('yes')
except ImportError:
    print('no')" 2>/dev/null)

if [ "$RKNN_INSTALLED" = "yes" ]; then
    print_success "RKNN Toolkit2 already installed in venv-rknn"
    deactivate
else
    print_info "RKNN Toolkit2 not found, installing..."
    
    # Check if rknn-toolkit2 directory is available
    if [ ! -d "rknn-toolkit2" ]; then
        print_error "rknn-toolkit2 directory not found in workspace"
        print_info "The directory should already be present. Please ensure rknn-toolkit2/ exists."
        deactivate
        exit 1
    fi
    
    print_info "Installing RKNN Toolkit2..."
    
    cd rknn-toolkit2/rknn-toolkit2
    
    # Detect Python version and select appropriate wheel
    PYTHON_VER=$($PYTHON_CMD -c 'import sys; print(f"cp{sys.version_info.major}{sys.version_info.minor}")')
    print_info "Detected Python version: $PYTHON_VER"
    
    # Find matching wheel file
    WHEEL_FILE=$(ls packages/x86_64/rknn_toolkit2*${PYTHON_VER}*.whl 2>/dev/null | head -1)
    
    if [ -z "$WHEEL_FILE" ]; then
        print_error "No matching RKNN Toolkit2 wheel found for Python version $PYTHON_VER"
        print_info "Available wheels:"
        ls packages/x86_64/rknn_toolkit2*.whl
        deactivate
        cd ../..
        exit 1
    fi
    
    # Find matching requirements file
    REQ_FILE=$(ls packages/x86_64/requirements_${PYTHON_VER}*.txt 2>/dev/null | head -1)
    
    print_info "Installing dependencies..."
    pip install --upgrade pip --quiet
    
    if [ ! -z "$REQ_FILE" ]; then
        print_info "Installing from $REQ_FILE..."
        pip install -r "$REQ_FILE"
    fi
    
    # Fix ONNX version compatibility issue with RKNN Toolkit 2.3.2
    print_info "Ensuring compatible ONNX version..."
    pip install "onnx>=1.16.1,<1.17.0" --force-reinstall
    
    print_info "Installing $WHEEL_FILE..."
    pip install "$WHEEL_FILE"
    
    cd ../..
    
    # Verify installation
    VERIFY=$($PYTHON_CMD -c "try:
    from rknn.api import RKNN
    print('success')
except ImportError as e:
    print(f'failed: {e}')")
    
    if [[ "$VERIFY" == "success" ]]; then
        print_success "RKNN Toolkit2 installation verified"
    else
        print_error "RKNN Toolkit2 installation failed: $VERIFY"
        deactivate
        exit 1
    fi
    
    deactivate
    print_success "RKNN environment setup complete"
fi

echo ""


# Download/Prepare Model

print_header "Step 3: Prepare Model"

MODEL_PT="${MODEL_NAME}.pt"

if [ ! -z "$MODEL_URL" ]; then
    print_info "Downloading model from $MODEL_URL..."
    wget -O "$MODEL_PT" "$MODEL_URL" || curl -o "$MODEL_PT" "$MODEL_URL"
    print_success "Model downloaded: $MODEL_PT"
elif [ ! -f "$MODEL_PT" ]; then
    print_info "Model file not found. Will download using ultralytics..."
    source venv-onnx/bin/activate
    $PYTHON_CMD -c "from ultralytics import YOLO; YOLO('${MODEL_NAME}.pt')"
    deactivate
    print_success "Model downloaded: $MODEL_PT"
else
    print_success "Using existing model: $MODEL_PT"
fi

echo ""


# Export PyTorch to ONNX at Target Resolution

print_header "Step 4: Export PyTorch to ONNX (${EXPORT_IMGSZ}x${EXPORT_IMGSZ})"

source venv-onnx/bin/activate

MODEL_ONNX="${MODEL_NAME}.onnx"

if [ -f "$MODEL_ONNX" ]; then
    print_warning "ONNX model already exists: $MODEL_ONNX"
    read -p "Overwrite? (y/n): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        print_info "Skipping ONNX export"
    else
        rm "$MODEL_ONNX"
        print_info "Exporting to ONNX (opset=$OPSET_VERSION, imgsz=$EXPORT_IMGSZ)..."
        yolo export model="$MODEL_PT" format=onnx opset=$OPSET_VERSION imgsz=$EXPORT_IMGSZ dynamic=False
        print_success "ONNX export complete: $MODEL_ONNX"
    fi
else
    print_info "Exporting to ONNX (opset=$OPSET_VERSION, imgsz=$EXPORT_IMGSZ)..."
    yolo export model="$MODEL_PT" format=onnx opset=$OPSET_VERSION imgsz=$EXPORT_IMGSZ dynamic=False
    print_success "ONNX export complete: $MODEL_ONNX"
fi

deactivate
echo ""


# Remove Nodes for YOLO11

print_header "Step 5: Remove DFL Nodes (YOLO11 specific)"

source venv-onnx/bin/activate

MODEL_MODIFIED="modified_${MODEL_NAME}_${EXPORT_IMGSZ}x${EXPORT_IMGSZ}.onnx"

if [ -f "$MODEL_MODIFIED" ]; then
    print_warning "Modified model already exists: $MODEL_MODIFIED"
    rm "$MODEL_MODIFIED"
fi

print_info "Removing DFL nodes using removenodes11n.py..."
$PYTHON_CMD utils/removenodes11n.py "$MODEL_ONNX" "$MODEL_MODIFIED"
print_success "Node removal complete: $MODEL_MODIFIED"

deactivate
echo ""

# Convert to RKNN (Direct Static Conversion)

print_header "Step 6: Convert to RKNN (Static ${GRAPH_SIZE})"

source venv-rknn/bin/activate

print_info "Converting to RKNN format (no dynamic shapes)..."
print_info "  Platform: $PLATFORM"
print_info "  Graph size: $GRAPH_SIZE"
print_info "  Quantization: $ENABLE_QUANTIZATION"

# Build command
RKNN_CMD="$PYTHON_CMD utils/onnx2rknn.py $MODEL_MODIFIED --platform $PLATFORM --graphsz $GRAPH_SIZE"

if [ "$ENABLE_QUANTIZATION" = true ]; then
    RKNN_CMD="$RKNN_CMD --quantize --dataset $DATASET_FILE"
fi

# Execute conversion
eval $RKNN_CMD

# Find the generated RKNN file
RKNN_FILE=$(ls -t modified_${MODEL_NAME}_${EXPORT_IMGSZ}x${EXPORT_IMGSZ}*.rknn 2>/dev/null | head -1)

if [ -z "$RKNN_FILE" ]; then
    print_error "RKNN file not generated"
    deactivate
    exit 1
fi

print_success "RKNN conversion complete: $RKNN_FILE"

deactivate
echo ""

# Summary

print_header "Conversion Complete!"

echo ""
echo "Generated files:"
echo "  1. ONNX (static ${EXPORT_IMGSZ}x${EXPORT_IMGSZ}): $MODEL_ONNX"
echo "  2. ONNX (modified):                  $MODEL_MODIFIED"
echo "  3. RKNN (final):                     $RKNN_FILE"
echo ""
echo "Note: This conversion uses static shapes (no dynamic conversion)."
echo "The model is exported at ${EXPORT_IMGSZ}x${EXPORT_IMGSZ} to ensure proper graph generation."
echo ""
echo "Next steps:"
echo "  - Test the model using utils/rknn_inference.py on RK35xx hardware"
echo "  - Deploy to your Vicharak Axon board"
echo ""
print_success "Pipeline completed successfully!"
