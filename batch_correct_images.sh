#!/bin/bash
# Batch process all LSMI test images with CAM-based correction

INPUT_DIR="Data/LSMI_Test_Package/images"
OUTPUT_DIR="visualizations/cam_correction_batch"
MODEL="standard"
CAM_METHOD="gradcam"

echo "Starting batch correction for all images..."
echo "Input directory: $INPUT_DIR"
echo "Output directory: $OUTPUT_DIR"
echo "Model: $MODEL"
echo "CAM method: $CAM_METHOD"
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Count total images
TOTAL=$(ls $INPUT_DIR/*.nef 2>/dev/null | wc -l)
echo "Found $TOTAL images to process"
echo ""

# Process counter
COUNT=0

# Process each NEF file
for img in $INPUT_DIR/*.nef; do
    COUNT=$((COUNT + 1))
    BASENAME=$(basename "$img" .nef)
    
    echo "[$COUNT/$TOTAL] Processing: $BASENAME"
    
    python src/display_scripts/correct_with_cam.py \
        --image "$img" \
        --model "$MODEL" \
        --cam "$CAM_METHOD" \
        --output "$OUTPUT_DIR" \
        2>&1 | grep -E "(Predicted class|Mapped|GT mask correction|Saved visualization|Error|Could not)"
    
    if [ $? -eq 0 ]; then
        echo "  ✓ Success"
    else
        echo "  ✗ Failed"
    fi
    echo ""
done

echo "Batch processing complete!"
echo "Results saved to: $OUTPUT_DIR"
echo "Total images processed: $COUNT"
