#!/bin/bash
# Quick test of PhysTwin on PGND custom cloth data

set -e

echo "🧪 Testing PhysTwin on Custom PGND Cloth Data"
echo "=============================================="
echo ""

# Configuration
PGND_EPISODE="/home/fashionista/pgnd/experiments/log/data_cloth/1224_cloth_fold_processed/episode_0000"
PHYSTWIN_DATA="/home/fashionista/PhysTwin/data/pgnd_cloth"
CASE_NAME="pgnd_cloth_ep0_test"

echo "📁 Step 1: Convert PGND data to PhysTwin format"
echo "   Episode: $PGND_EPISODE"
echo "   Output: $PHYSTWIN_DATA/$CASE_NAME"
echo ""

python convert_pgnd_to_phystwin.py \
  --pgnd_episode "$PGND_EPISODE" \
  --output_dir "$PHYSTWIN_DATA" \
  --case_name "$CASE_NAME"

echo ""
echo "✅ Conversion complete!"
echo ""
echo "📊 Next steps:"
echo ""
echo "1️⃣  Data Processing (generates tracking):"
echo "   cd /home/fashionista/PhysTwin"
echo "   # TODO: Need to create single-case processing script"
echo ""
echo "2️⃣  Physics Optimization (25 min):"
echo "   cd /home/fashionista/PhysTwin"
echo "   python train_warp.py \\"
echo "     --base_path $PHYSTWIN_DATA \\"
echo "     --case_name $CASE_NAME \\"
echo "     --train_frame 80"
echo ""
echo "3️⃣  Future Prediction Testing:"
echo "   python inference_warp.py \\"
echo "     --base_path $PHYSTWIN_DATA \\"
echo "     --case_name $CASE_NAME"
echo ""
echo "4️⃣  Compare with PGND:"
echo "   python compare_predictions.py \\"
echo "     --pgnd_episode $PGND_EPISODE \\"
echo "     --phystwin_case $CASE_NAME"
echo ""
echo "🎯 Goal: Test if 25-min PhysTwin optimization on your cloth"
echo "   can match or beat PGND's learned dynamics!"
