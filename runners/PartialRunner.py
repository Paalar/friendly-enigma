from models.multiTaskLearner import MultiTaskLearner
from models.singleTaskLearner import SingleTaskLearner
from data.partialDataModule import get_partial_data_module
from runners.STL_runner import STLRunner
from config import config
from data import (
    stlGmscDataModule,
    gmscDataModule,
    stlFakeDataModule,
    mtlFakeDataModule,
    helocDataModule,
    cchvaeDataModule,
)

alternatives = {
    "stl-heloc": helocDataModule.HelocDataModule,
    "mtl-heloc": cchvaeDataModule.CchvaeDataModule,
    "mtl-gmsc": gmscDataModule.GMSCDataModule,
    "stl-gmsc": stlGmscDataModule.STLGMSCDataModule,
    "stl-fake": stlFakeDataModule.STLFakeDataModule,
    "mtl-fake": mtlFakeDataModule.MTLFakeDataModule,
}


class PartialRunner(STLRunner):
    def __init__(self, **kwargs):
        if not "args" in kwargs:
            raise Exception(
                "Did not receive any arguments, 'partial' runner has no defaults."
            )

        arguments = kwargs["args"]
        if not "train_size" in arguments or not "module_type" in arguments:
            raise Exception(
                "train_size and module_type need to be set when using 'partial' runner."
            )

        module_type = arguments.module_type
        module = alternatives[module_type]
        print("Running with module: ", module_type)
        partial_module = get_partial_data_module(
            module, training_size_percentage=arguments.train_size
        )
        model_type = module_type.split("-")[0]
        model = None
        output_length = None
        if model_type == "stl":
            model = SingleTaskLearner
        elif model_type == "mtl":
            model = MultiTaskLearner
        else:
            raise Exception(
                f"Could not parse model type from module type {module_type}"
            )

        cli_args = kwargs["args"] if "args" in kwargs else None
        use_signloss = cli_args.use_signloss if cli_args else False
        super().__init__(
            **kwargs,
            data_module=partial_module(),
            max_epochs=config["mtl_epochs"],
            checkpoints_prefix=f"{module_type}-partial",
        )
        output_length = (1, len(self.data_module.labels)) if model_type == "mtl" else 1
        self.model = model(
            model_core=self.model_core,
            input_length=self.nodes_before_split,
            output_length=output_length,
            use_signloss=use_signloss,
        )
