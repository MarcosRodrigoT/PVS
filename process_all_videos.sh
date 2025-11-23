#!/bin/bash

# ============================================================================
# Process Multiple Videos Script
# ============================================================================
# This script runs the PVS pipeline on multiple videos in FOUR modes:
# 1. SportCLIP WITH --use-updated-target
# 2. SportCLIP WITHOUT --use-updated-target
# 3. QD-DETR WITH --use-updated-target
# 4. QD-DETR WITHOUT --use-updated-target
#
# Simply uncomment/comment the videos you want to process below
# ============================================================================

# Note: NOT using 'set -e' so the script continues even if one mode fails

# ============================================================================
# VIDEO CONFIGURATION - Edit this list to select which videos to process
# ============================================================================
# Format: "video_name:sportclip_sentences:qddetr_query"
# Comment out (with #) videos you don't want to process

VIDEOS=(
    # 100 meter videos (commented out by default)
    # "100m_video1:data/sentences/100m.json:data/sentences/QD-DETR/100m.txt"
    # "100m_video2:data/sentences/100m.json:data/sentences/QD-DETR/100m.txt"
    # "100m_video3:data/sentences/100m.json:data/sentences/QD-DETR/100m.txt"
    # "100m_video4:data/sentences/100m.json:data/sentences/QD-DETR/100m.txt"
    # "100m_video5:data/sentences/100m.json:data/sentences/QD-DETR/100m.txt"

    # High jump videos
    "highjump_video1:data/sentences/high_jump.json:data/sentences/QD-DETR/high_jump.txt"
    "highjump_video2:data/sentences/high_jump.json:data/sentences/QD-DETR/high_jump.txt"
    "highjump_video3:data/sentences/high_jump.json:data/sentences/QD-DETR/high_jump.txt"
    "highjump_video4:data/sentences/high_jump.json:data/sentences/QD-DETR/high_jump.txt"
    "highjump_video5:data/sentences/high_jump.json:data/sentences/QD-DETR/high_jump.txt"

    # Javelin videos
    "javelin_video1:data/sentences/javelin.json:data/sentences/QD-DETR/javelin.txt"
    "javelin_video2:data/sentences/javelin.json:data/sentences/QD-DETR/javelin.txt"
    "javelin_video3:data/sentences/javelin.json:data/sentences/QD-DETR/javelin.txt"
    "javelin_video4:data/sentences/javelin.json:data/sentences/QD-DETR/javelin.txt"
    "javelin_video5:data/sentences/javelin.json:data/sentences/QD-DETR/javelin.txt"

    # Long jump videos
    "longjump_video1:data/sentences/long_jump.json:data/sentences/QD-DETR/long_jump.txt"
    "longjump_video2:data/sentences/long_jump.json:data/sentences/QD-DETR/long_jump.txt"
    "longjump_video3:data/sentences/long_jump.json:data/sentences/QD-DETR/long_jump.txt"
    "longjump_video4:data/sentences/long_jump.json:data/sentences/QD-DETR/long_jump.txt"
    "longjump_video5:data/sentences/long_jump.json:data/sentences/QD-DETR/long_jump.txt"

    # Pole vault videos
    "polevault_video1:data/sentences/pole_vault.json:data/sentences/QD-DETR/pole_vault.txt"
    "polevault_video2:data/sentences/pole_vault.json:data/sentences/QD-DETR/pole_vault.txt"
    "polevault_video3:data/sentences/pole_vault.json:data/sentences/QD-DETR/pole_vault.txt"
    "polevault_video4:data/sentences/pole_vault.json:data/sentences/QD-DETR/pole_vault.txt"
    "polevault_video5:data/sentences/pole_vault.json:data/sentences/QD-DETR/pole_vault.txt"
)

# ============================================================================
# SCRIPT CONFIGURATION
# ============================================================================
SMOOTH_WINDOW=200

# ============================================================================
# DO NOT EDIT BELOW THIS LINE
# ============================================================================

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Log file
LOG_FILE="processing_log_$(date +%Y%m%d_%H%M%S).txt"

echo -e "${BLUE}============================================================================${NC}"
echo -e "${BLUE}PVS Multi-Video Processing Script${NC}"
echo -e "${BLUE}============================================================================${NC}"
echo ""
echo "Log file: $LOG_FILE"
echo ""

# Display videos to process
echo -e "${YELLOW}Videos to process:${NC}"
for entry in "${VIDEOS[@]}"; do
    IFS=':' read -r video_name sportclip_sentences qddetr_query <<< "$entry"
    echo "  - $video_name"
    echo "      SportCLIP: $sportclip_sentences"
    echo "      QD-DETR:   $qddetr_query"
done
echo ""
echo "Total: ${#VIDEOS[@]} videos (4 modes each = $((${#VIDEOS[@]} * 4)) runs)"
echo ""

# Ask for confirmation
read -p "Do you want to proceed? (y/n) " -n 1 -r
echo ""
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Processing cancelled."
    exit 0
fi

echo ""
echo -e "${BLUE}Starting batch processing...${NC}"
echo "Start time: $(date)" | tee -a "$LOG_FILE"
echo ""

# Counters
successful=0
failed=0
failed_videos=()

# Process each video
for entry in "${VIDEOS[@]}"; do
    IFS=':' read -r video_name sportclip_sentences qddetr_query <<< "$entry"

    echo -e "${BLUE}============================================================================${NC}"
    echo -e "${BLUE}Processing: ${video_name}${NC}"
    echo -e "${BLUE}============================================================================${NC}"
    echo "SportCLIP sentences: $sportclip_sentences"
    echo "QD-DETR query file:  $qddetr_query"
    echo ""

    # Check if video file exists
    if [[ ! -f "data/${video_name}.mp4" ]]; then
        echo -e "${RED}✗ Video file not found: data/${video_name}.mp4${NC}" | tee -a "$LOG_FILE"
        echo -e "${YELLOW}  Skipping...${NC}"
        ((failed++))
        failed_videos+=("$video_name (file not found)")
        echo ""
        continue
    fi

    # Read QD-DETR query from file
    if [[ ! -f "$qddetr_query" ]]; then
        echo -e "${RED}✗ QD-DETR query file not found: $qddetr_query${NC}" | tee -a "$LOG_FILE"
        echo -e "${YELLOW}  Skipping...${NC}"
        ((failed++))
        failed_videos+=("$video_name (query file not found)")
        echo ""
        continue
    fi
    qddetr_query_text=$(cat "$qddetr_query" | tr -d '\n')

    # Track if any mode fails for this video
    video_failed=false

    # ========== Mode 1: SportCLIP WITH --use-updated-target ==========
    echo -e "${GREEN}>>> Mode 1/4: SportCLIP WITH --use-updated-target${NC}"
    echo "Command: python main.py --video $video_name --video-summarizer sportclip --sentences-file $sportclip_sentences --smooth-window $SMOOTH_WINDOW --use-updated-target"
    echo ""

    if python main.py \
        --video "$video_name" \
        --video-summarizer sportclip \
        --sentences-file "$sportclip_sentences" \
        --smooth-window "$SMOOTH_WINDOW" \
        --use-updated-target 2>&1 | tee -a "$LOG_FILE"; then
        echo -e "${GREEN}✓ Mode 1/4 completed successfully${NC}" | tee -a "$LOG_FILE"
    else
        echo -e "${RED}✗ Mode 1/4 failed for $video_name${NC}" | tee -a "$LOG_FILE"
        ((failed++))
        failed_videos+=("$video_name (SportCLIP with updated)")
        video_failed=true
    fi

    echo ""

    # ========== Mode 2: SportCLIP WITHOUT --use-updated-target ==========
    echo -e "${GREEN}>>> Mode 2/4: SportCLIP WITHOUT --use-updated-target${NC}"
    echo "Command: python main.py --video $video_name --video-summarizer sportclip --sentences-file $sportclip_sentences --smooth-window $SMOOTH_WINDOW"
    echo ""

    if python main.py \
        --video "$video_name" \
        --video-summarizer sportclip \
        --sentences-file "$sportclip_sentences" \
        --smooth-window "$SMOOTH_WINDOW" 2>&1 | tee -a "$LOG_FILE"; then
        echo -e "${GREEN}✓ Mode 2/4 completed successfully${NC}" | tee -a "$LOG_FILE"
    else
        echo -e "${RED}✗ Mode 2/4 failed for $video_name${NC}" | tee -a "$LOG_FILE"
        ((failed++))
        failed_videos+=("$video_name (SportCLIP without updated)")
        video_failed=true
    fi

    echo ""

    # ========== Mode 3: QD-DETR WITH --use-updated-target ==========
    echo -e "${GREEN}>>> Mode 3/4: QD-DETR WITH --use-updated-target${NC}"
    echo "Command: python main.py --video $video_name --video-summarizer qddetr --qddetr-query \"$qddetr_query_text\" --smooth-window $SMOOTH_WINDOW --use-updated-target"
    echo ""

    if python main.py \
        --video "$video_name" \
        --video-summarizer qddetr \
        --qddetr-query "$qddetr_query_text" \
        --smooth-window "$SMOOTH_WINDOW" \
        --use-updated-target 2>&1 | tee -a "$LOG_FILE"; then
        echo -e "${GREEN}✓ Mode 3/4 completed successfully${NC}" | tee -a "$LOG_FILE"
    else
        echo -e "${RED}✗ Mode 3/4 failed for $video_name${NC}" | tee -a "$LOG_FILE"
        ((failed++))
        failed_videos+=("$video_name (QD-DETR with updated)")
        video_failed=true
    fi

    echo ""

    # ========== Mode 4: QD-DETR WITHOUT --use-updated-target ==========
    echo -e "${GREEN}>>> Mode 4/4: QD-DETR WITHOUT --use-updated-target${NC}"
    echo "Command: python main.py --video $video_name --video-summarizer qddetr --qddetr-query \"$qddetr_query_text\" --smooth-window $SMOOTH_WINDOW"
    echo ""

    if python main.py \
        --video "$video_name" \
        --video-summarizer qddetr \
        --qddetr-query "$qddetr_query_text" \
        --smooth-window "$SMOOTH_WINDOW" 2>&1 | tee -a "$LOG_FILE"; then
        echo -e "${GREEN}✓ Mode 4/4 completed successfully${NC}" | tee -a "$LOG_FILE"
    else
        echo -e "${RED}✗ Mode 4/4 failed for $video_name${NC}" | tee -a "$LOG_FILE"
        ((failed++))
        failed_videos+=("$video_name (QD-DETR without updated)")
        video_failed=true
    fi

    echo ""

    # Update success counter only if all modes succeeded
    if [ "$video_failed" = false ]; then
        echo -e "${GREEN}✓✓✓✓ All 4 modes completed for $video_name${NC}" | tee -a "$LOG_FILE"
        ((successful++))
    else
        echo -e "${YELLOW}⚠ Some modes failed for $video_name (see above)${NC}" | tee -a "$LOG_FILE"
    fi

    # Progress update
    echo ""
    echo -e "${BLUE}----------------------------------------------------------------------------${NC}"
    echo -e "${BLUE}Progress: $((successful + failed))/${#VIDEOS[@]} videos processed${NC}"
    echo -e "${GREEN}Successful (all 4 modes): $successful${NC}"
    echo -e "${RED}With failures: $failed${NC}"
    echo -e "${BLUE}----------------------------------------------------------------------------${NC}"
    echo ""
done

# Final summary
echo ""
echo -e "${BLUE}============================================================================${NC}"
echo -e "${BLUE}BATCH PROCESSING COMPLETE${NC}"
echo -e "${BLUE}============================================================================${NC}"
echo "End time: $(date)" | tee -a "$LOG_FILE"
echo ""
echo "Summary:" | tee -a "$LOG_FILE"
echo "  Total videos: ${#VIDEOS[@]}" | tee -a "$LOG_FILE"
echo -e "  ${GREEN}Successful: $successful${NC}" | tee -a "$LOG_FILE"
echo -e "  ${RED}Failed: $failed${NC}" | tee -a "$LOG_FILE"

if [ $failed -gt 0 ]; then
    echo "" | tee -a "$LOG_FILE"
    echo "Failed videos:" | tee -a "$LOG_FILE"
    for vid in "${failed_videos[@]}"; do
        echo "  - $vid" | tee -a "$LOG_FILE"
    done
fi

echo ""
echo "Full log saved to: $LOG_FILE"
echo ""

# Exit with error code if any videos failed
if [ $failed -gt 0 ]; then
    exit 1
else
    echo -e "${GREEN}All videos processed successfully!${NC}"
    exit 0
fi
