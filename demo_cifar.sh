python feat_extract.py --in-dataset CIFAR-10  --out-datasets SVHN LSUN iSUN dtd places365 --name resnet18-supcon  --model-arch resnet18-supcon
python run_cifar.py --in-dataset CIFAR-10  --out-datasets SVHN LSUN iSUN dtd places365 --name resnet18-supcon  --model-arch resnet18-supcon

python feat_extract.py --in-dataset CIFAR-10  --out-datasets LSUN_FIX ImageNet_FIX Imagenet_resize CIFAR-100 --name resnet18-supcon  --model-arch resnet18-supcon
python run_cifar.py --in-dataset CIFAR-10  --out-datasets LSUN_FIX ImageNet_FIX Imagenet_resize CIFAR-100 --name resnet18-supcon  --model-arch resnet18-supcon



python feat_extract.py --in-dataset CIFAR-100  --out-datasets SVHN LSUN iSUN dtd places365 --name resnet34-supcon2  --model-arch resnet34-supcon
python run_cifar.py --in-dataset CIFAR-100  --out-datasets SVHN LSUN iSUN dtd places365 --name resnet34-supcon2  --model-arch resnet34-supcon
