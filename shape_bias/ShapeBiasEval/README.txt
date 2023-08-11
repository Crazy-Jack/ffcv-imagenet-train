1. Clone the original modelvshuman repo
2. Put this folder under "/model-vs-human/examples" (actually it should be able to put under any location within the repo, just to make sure any other problem wouldn't show up)
3. Make sure the current env is able to run modelvshuman (remember to run export MODELVSHUMANDIR=/absolute/path/to/this/repository/)
4. Use the following code to get the shape score, remember to change '/path/to/shape_bias' to the path of this folder

from torchvision import models
import sys

sys.path.append('/path/to/shape_bias')

from test_evalute_shape import run_evaluation


from torchvision import models

vgg16 = models.vgg16(pretrained=True)
result = ShapeBiasEval.run_evaluation(vgg16)
print(result)

5. Here is an example of the output:
[('airplane', 0.016666666666666666), ('bear', 0.046875), ('bicycle', 0.043478260869565216), ('bird', 0.0), ('boat', 0.0), ('bottle', 0.08333333333333333), ('car', 0.25757575757575757), ('cat', 0.078125), ('chair', 0.045454545454545456), ('clock', 0.2786885245901639), ('dog', 0.08928571428571429), ('elephant', 0.016666666666666666), ('keyboard', 0.03125), ('knife', 0.0), ('oven', 0.046875), ('truck', 0.1911764705882353)]