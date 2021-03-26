from models.multiTaskLearner import MultiTaskLearner

classifier = MultiTaskLearner.load_from_checkpoint("./data/processing/heloc-epoch=07-loss_validate=8.05.ckpt")

l = classifier.named_parameters()
for name, value in l:
    print(name)
# for parameter in classifier.parameters():
#     print(f"{parameter}= Gradient={parameter.grad}")

