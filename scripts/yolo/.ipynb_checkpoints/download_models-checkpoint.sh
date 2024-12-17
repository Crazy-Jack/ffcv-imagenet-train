
export CURRENT_DIR=$PWD
cd /workspace

# download models 
EXP_NAME=$1
FILENAME=$2

if [ -e "/workspace/b2" ]; then
    echo "=> /workspace/b2 existed!"
else
    cd /workspace
    wget https://github.com/Backblaze/B2_Command_Line_Tool/releases/latest/download/b2-linux
    mv b2-linux b2
    chmod +x b2
    # authentica the b2
    # /workspace/b2 account authorize 
fi


mkdir -p /workspace/model_store/$EXP_NAME

if [ -e "/workspace/model_store/$EXP_NAME/$FILENAME" ]; then
    echo "/workspace/model_store/$EXP_NAME/$FILENAME existed!"
else
/workspace/b2 file download b2://model-store/$EXP_NAME/$FILENAME /workspace/model_store/$EXP_NAME/$FILENAME
fi


echo "Check Model Downloaded:)"
ls -sh /workspace/model_store/$EXP_NAME/$FILENAME


