
export CURRENT_DIR=$PWD


# download models 
EXP_NAME=$1
FILENAME=$2
WORK_DIR=$3

cd $WORK_DIR
if [ -e "$WORK_DIR/b2" ]; then
    echo "=> $WORK_DIR/b2 existed!"
else
    cd $WORK_DIR
    wget https://github.com/Backblaze/B2_Command_Line_Tool/releases/latest/download/b2-linux
    mv b2-linux b2
    chmod +x b2
    # authentica the b2
    # /workspace/b2 account authorize 
fi


mkdir -p $WORK_DIR/model_store/$EXP_NAME

if [ -e "$WORK_DIR/model_store/$EXP_NAME/$FILENAME" ]; then
    echo "$WORK_DIR/model_store/$EXP_NAME/$FILENAME existed!"
else
$WORK_DIR/b2 file download b2://model-store/$EXP_NAME/$FILENAME $WORK_DIR/model_store/$EXP_NAME/$FILENAME
fi


echo "Check Model Downloaded:)"
ls -sh $WORK_DIR/model_store/$EXP_NAME/$FILENAME


