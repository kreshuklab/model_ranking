import yaml
from pathlib import Path
from typing import (
    Literal,
    Optional,
    Sequence,
    Tuple,
    Dict,
    List,
    Union,
    Any,
    Mapping,
    assert_never,
)
from model_ranking.dataclass import (
    ConsistencyConfig,
    Eval_TIF_DataloaderMetaConfig,
    EvalDataloaderMetaConfig,
    EvalSB1410DataloaderMetaConfig,
    EvaluateConfig,
    # Pytorch3DUnetLoaderConfig,
    Pytorch3DUnetModelConfig,
    SBIAD1410LoaderMetaConfig,
    SummaryResultsConfig,
    WandbConfig,
    # TIFPredictionLoadersConfig,
    FeatureNoisePerturbationConfig,
    FeatureDropPerturbationConfig,
    DropOutPerturbationConfig,
    MetaConfig,
    InputGaussianConfig,
    InputPerturbationConfig,
    Pytorch3DUnetLoaderMetaConfig,
)
from model_ranking.utils import get_output_dir

from pytorch3dunet.unet3d.config import load_config_direct  # type: ignore


def tuple_representer(dumper: Any, data: Any) -> Any:
    return dumper.represent_sequence(
        "tag:yaml.org,2002:seq", list(data), flow_style=True
    )


class NoAliasDumper(yaml.SafeDumper):
    def ignore_aliases(self, data: Any) -> bool:
        return True


def save_yaml(
    yaml_order: Sequence[Mapping[str, Any]],
    yaml_path: Path,
    overwrite: bool = False,
) -> None:
    # check yaml file of same name doesn't exist in location
    if yaml_path.exists():
        if overwrite:
            print(f"Overwriting yaml file at {yaml_path}")
            # delete existing yaml file
            yaml_path.unlink()
        else:
            # if exists print warning and skip saving yaml file
            print(f"Yaml file {yaml_path} already exists, skipping save")
            return

    print(f"Saving yaml file to {yaml_path}")
    # create path if it doesn't exist
    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    with open(yaml_path, "w") as yaml_file:
        yaml.add_representer(tuple, tuple_representer, Dumper=NoAliasDumper)
        for yaml_dict in yaml_order:
            _ = yaml_file.write(
                yaml.dump(
                    yaml_dict,
                    default_flow_style=False,
                    sort_keys=False,
                    Dumper=NoAliasDumper,
                )
            )
            _ = yaml_file.write("\n")


def generate_aug_config(
    aug_strengths: Dict[str, List[Tuple[float, float]]],
    aug_abbr: Dict[str, str] = {
        "brt": "RandomBrightness",
        "ctr": "RandomContrast",
        "gamma": "RandomGamma",
        "gauss": "AdditiveGaussianNoise",
    },
) -> Dict[str, Optional[Union[InputGaussianConfig, InputPerturbationConfig]]]:
    augs: Dict[str, Optional[Union[InputGaussianConfig, InputPerturbationConfig]]] = {}
    for aug_type in aug_strengths.keys():
        if aug_type == "none":
            aug_key = "none"
            augs[aug_key] = None
        else:
            for alpha_range in aug_strengths[aug_type]:
                aug_key = (
                    f"{aug_type}_a{str(alpha_range[0]).replace('.', '')}-"
                    f"{str(alpha_range[1]).replace('.', '')}"
                )
                if aug_type == "gauss":
                    augs[aug_key] = InputGaussianConfig(
                        name=aug_abbr[aug_type],
                        execution_probability=1,
                        scale=alpha_range,
                    )

                else:
                    augs[aug_key] = InputPerturbationConfig(
                        name=aug_abbr[aug_type],
                        execution_probability=1,
                        alpha=alpha_range,
                        clip_kwargs=None,
                    )
    return augs


def get_model_path(
    source_data: str,
    model_name: str,
    base_dir_path: str,
    seg_mode: Literal["instance", "semantic"] = "semantic",
) -> str:
    """Find path to model checkpoint

    Args:
        source_data (str): source dataset name
        model_name (str): source model name
        base_dir_path (str): path to directory contatining model checkpoints
        seg_mode (Literal[&quot;instance&quot;, &quot;semantic&quot;], optional): semantic or instance segmentation mode. Defaults to "semantic".

    Returns:
        str: = path to model checkpoint
    """
    if seg_mode == "instance":
        base_dir = Path(base_dir_path) / "Instance"

    else:
        base_dir = Path(base_dir_path) / "Semantic"

    model_paths = list(
        base_dir.glob(f"{source_data}/**/" + f"{model_name}/best_checkpoint.pytorch")
    )
    assert (
        len(model_paths) == 1
    ), f"number of path found = {len(model_paths)}, model ambiguous"
    model_path = str(model_paths[0])
    return model_path


DATASET_TO_MODEL_ABBREVIATIONS = {
    #### Nuclei
    "BBBC039": "BC",
    "DSB2018": "DSB",
    "Go-Nuclear": "GN",
    "HeLaNuc": "HN",
    "Hoechst": "Hst",
    "S_BIAD634": "634",
    "S_BIAD895": "895",
    "S_BIAD1196": "1196",
    "S_BIAD1410": "1410",
    #### Cells
    "FlyWing": "fw",
    "Ovules": "ov",
    "PNAS": "p",
    #### Mitochondria
    "EPFL": "E",
    "Hmito": "Hm",
    "Rmito": "Rm",
    "VNC": "V",
}

FEATURE_PERTURBATION_ABBREVIATIONS: Dict[str, str] = {
    "DropOutPerturbation": "DO",
    "FeatureDropPerturbation": "FD",
    "FeatureNoisePerturbation": "FN",
    "None": "none",
}


def generate_run_yamls(config_path: Union[str, Path]) -> Dict[str, List[Path]]:
    config, _ = load_config_direct(config_path)
    meta_cfg = MetaConfig.model_validate(config)
    yaml_paths: Dict[str, List[Path]] = {}
    for source_model in meta_cfg.source_models:
        source_model_path = get_model_path(
            source_data=source_model.source_name,
            model_name=source_model.model_name,
            base_dir_path=meta_cfg.model_dir_path,
            seg_mode=meta_cfg.segmentation_mode,
        )
        feat_pert_cfg = meta_cfg.feature_perturbations
        model_cfgs: Dict[str, Pytorch3DUnetModelConfig] = {}
        for feature_perturbation in feat_pert_cfg.perturbation_types:
            feature_abbrev = FEATURE_PERTURBATION_ABBREVIATIONS[feature_perturbation]
            if feature_perturbation != "None":
                if feature_perturbation == "DropOutPerturbation":
                    assert (
                        feat_pert_cfg.dropOut_rates is not None
                    ), "dropOut rates not provided"
                    assert (
                        feat_pert_cfg.spatial_dropout is not None
                    ), "spatial dropout not provided"
                    for dropOut_rate in feat_pert_cfg.dropOut_rates:
                        feature_perturbation_config = DropOutPerturbationConfig(
                            name=feature_perturbation,
                            random_seed=feat_pert_cfg.random_seed,
                            layers=feat_pert_cfg.layers,
                            drop_rate=dropOut_rate,
                            spatial_dropout=feat_pert_cfg.spatial_dropout,
                        )
                        feature_str = f"_a{str(dropOut_rate).replace('.','')}"
                        model_cfgs[feature_abbrev + feature_str] = (
                            source_model.create_config(
                                feature_perturbation=feature_perturbation_config
                            )
                        )

                elif feature_perturbation == "FeatureDropPerturbation":
                    assert (
                        feat_pert_cfg.featureDrop_thresholds is not None
                    ), "featureDrop thresholds not provided"
                    for featureDrop_th in feat_pert_cfg.featureDrop_thresholds:
                        feature_perturbation_config = FeatureDropPerturbationConfig(
                            name=feature_perturbation,
                            random_seed=feat_pert_cfg.random_seed,
                            layers=feat_pert_cfg.layers,
                            lower_th=featureDrop_th[0],
                            upper_th=featureDrop_th[1],
                        )
                        feature_str = f"_a{str(featureDrop_th[0]).replace('.','')}-{str(featureDrop_th[1]).replace('.','')}"
                        model_cfgs[feature_abbrev + feature_str] = (
                            source_model.create_config(
                                feature_perturbation=feature_perturbation_config
                            )
                        )
                elif feature_perturbation == "FeatureNoisePerturbation":
                    assert (
                        feat_pert_cfg.featureNoise_ranges is not None
                    ), "featureNoise ranges not provided"
                    for featureNoise_range in feat_pert_cfg.featureNoise_ranges:
                        feature_perturbation_config = FeatureNoisePerturbationConfig(
                            name=feature_perturbation,
                            random_seed=feat_pert_cfg.random_seed,
                            layers=feat_pert_cfg.layers,
                            uniform_range=featureNoise_range,
                        )

                        feature_str = f"_a{str(featureNoise_range).replace('.','')}"
                        model_cfgs[feature_abbrev + feature_str] = (
                            source_model.create_config(
                                feature_perturbation=feature_perturbation_config
                            )
                        )
                else:
                    assert_never(feature_perturbation)
            else:
                feature_name = feature_abbrev
                model_cfgs[feature_name] = source_model.create_config(
                    feature_perturbation=None
                )

        for target_cfg in meta_cfg.target_datasets:
            transfer_title = f"{source_model.source_name}_to_{target_cfg.name}"
            transfer_title_abbrev = (
                DATASET_TO_MODEL_ABBREVIATIONS[source_model.source_name]
                + "to"
                + DATASET_TO_MODEL_ABBREVIATIONS[target_cfg.name]
            )

            if isinstance(
                target_cfg.loader, Pytorch3DUnetLoaderMetaConfig
            ) or isinstance(target_cfg.loader, SBIAD1410LoaderMetaConfig):
                if target_cfg.loader.global_percentiles is not None:
                    percentiles_save_name = (
                        f'{str(target_cfg.loader.global_percentiles[0]).replace(".", "")}_'
                        f'{str(target_cfg.loader.global_percentiles[1]).replace(".", "")}'
                    )
                else:
                    percentiles_save_name = "Normalize"
            else:
                if target_cfg.loader.percentiles is not None:
                    percentiles_save_name = (
                        f'{str(target_cfg.loader.percentiles[0]).replace(".", "")}_'
                        f'{str(target_cfg.loader.percentiles[1]).replace(".", "")}'
                    )
                else:
                    percentiles_save_name = "Normalize"

            output_folder_path = get_output_dir(
                source=source_model.source_name,
                target=target_cfg.name,
                model_name=source_model.model_name,
                output_folder=f"norm_{percentiles_save_name}",
                approach=meta_cfg.output_settings.approach,
                result_type=meta_cfg.output_settings.result_dir,
                base_seg_folder=meta_cfg.output_settings.base_dir_path,
            )

            augs_cfg = generate_aug_config(meta_cfg.input_augs)
            for feature_perturbation_name, model_cfg in model_cfgs.items():
                for aug_name in augs_cfg.keys():
                    if (feature_perturbation_name == "none") & (aug_name == "none"):
                        save_name = "none"
                    elif (feature_perturbation_name == "none") & (aug_name != "none"):
                        save_name = aug_name
                    elif (feature_perturbation_name != "none") & (aug_name == "none"):
                        save_name = feature_perturbation_name
                    else:
                        save_name = f"feat_{feature_perturbation_name}_aug_{aug_name}"

                    pred_dir_path = (
                        output_folder_path + "/" + save_name + "/predictions"
                    )
                    none_pred_path = output_folder_path + "/" + "none" + "/predictions"
                    # make directory if needed
                    Path(pred_dir_path).mkdir(parents=True, exist_ok=True)

                    # Get Predictor config
                    if meta_cfg.segmentation_mode == "semantic":
                        assert (
                            target_cfg.eval_dataloader_semantic is not None
                        ), f"Eval dataloader semantic is None for for selected mode == {meta_cfg.segmentation_mode}"
                        project_name = f"{source_model.source_name}_predictions"
                        predictor_cfg = target_cfg.predictor_semantic
                        if (
                            target_cfg.eval_dataloader_semantic.name
                            == "StandardEvalDataset"
                        ):
                            eval_loader_cfg = (
                                target_cfg.eval_dataloader_semantic.create_config(
                                    aug_name=aug_name,
                                    pred_path=pred_dir_path,
                                    data_base_path=meta_cfg.data_base_path,
                                )
                            )
                            assert isinstance(
                                target_cfg.consis_dataloader_semantic,
                                EvalDataloaderMetaConfig,
                            )
                            consis_loader_cfg = target_cfg.consis_dataloader_semantic.create_consis_config(
                                aug_name=aug_name,
                                perturbed_path=(pred_dir_path,),
                                unperturbed_path=(none_pred_path,),
                                data_base_path=meta_cfg.data_base_path,
                            )
                        elif (
                            target_cfg.eval_dataloader_semantic.name
                            == "S_BIAD1410_Dataset"
                        ):
                            eval_loader_cfg = (
                                target_cfg.eval_dataloader_semantic.create_config(
                                    img_paths=(pred_dir_path,),
                                    data_base_path=meta_cfg.data_base_path,
                                )
                            )
                            assert isinstance(
                                target_cfg.consis_dataloader_semantic,
                                EvalSB1410DataloaderMetaConfig,
                            )
                            consis_loader_cfg = target_cfg.consis_dataloader_semantic.create_consis_config(
                                perturbed_paths=(pred_dir_path,),
                                unperturbed_paths=(none_pred_path,),
                            )

                        else:
                            eval_loader_cfg = (
                                target_cfg.eval_dataloader_semantic.create_config(
                                    image_dir=(pred_dir_path,),
                                    data_base_path=meta_cfg.data_base_path,
                                )
                            )

                            if (
                                target_cfg.consis_dataloader_semantic.name
                                == "TIF_txt_Dataset"
                            ):
                                consis_loader_cfg = target_cfg.consis_dataloader_semantic.create_consis_config(
                                    perturbed_dir=(pred_dir_path,),
                                    unperturbed_dir=(none_pred_path,),
                                    data_base_path=meta_cfg.data_base_path,
                                )
                            else:
                                assert isinstance(
                                    target_cfg.consis_dataloader_semantic,
                                    Eval_TIF_DataloaderMetaConfig,
                                )
                                consis_loader_cfg = target_cfg.consis_dataloader_semantic.create_consis_config(
                                    perturbed_dir=(pred_dir_path,),
                                    unperturbed_dir=(none_pred_path,),
                                )

                    elif meta_cfg.segmentation_mode == "instance":
                        assert (
                            target_cfg.eval_dataloader_instance is not None
                        ), f"Eval dataloader instance is None for for selected mode == {meta_cfg.segmentation_mode}"
                        project_name = f"{source_model.source_name}_IN_predictions"
                        predictor_cfg = target_cfg.predictor_instance
                        if (
                            target_cfg.eval_dataloader_instance.name
                            == "StandardEvalDataset"
                        ):
                            eval_loader_cfg = (
                                target_cfg.eval_dataloader_instance.create_config(
                                    aug_name=aug_name,
                                    pred_path=pred_dir_path,
                                    data_base_path=meta_cfg.data_base_path,
                                )
                            )
                            assert isinstance(
                                target_cfg.consis_dataloader_instance,
                                EvalDataloaderMetaConfig,
                            )
                            consis_loader_cfg = target_cfg.consis_dataloader_instance.create_consis_config(
                                aug_name=aug_name,
                                perturbed_path=(pred_dir_path,),
                                unperturbed_path=(none_pred_path,),
                                data_base_path=meta_cfg.data_base_path,
                            )
                        elif (
                            target_cfg.eval_dataloader_instance.name
                            == "S_BIAD1410_Dataset"
                        ):
                            eval_loader_cfg = (
                                target_cfg.eval_dataloader_instance.create_config(
                                    img_paths=(pred_dir_path,),
                                    data_base_path=meta_cfg.data_base_path,
                                )
                            )
                            assert isinstance(
                                target_cfg.consis_dataloader_instance,
                                EvalSB1410DataloaderMetaConfig,
                            )
                            consis_loader_cfg = target_cfg.consis_dataloader_instance.create_consis_config(
                                perturbed_paths=(pred_dir_path,),
                                unperturbed_paths=(none_pred_path,),
                            )
                        else:
                            eval_loader_cfg = (
                                target_cfg.eval_dataloader_instance.create_config(
                                    image_dir=(pred_dir_path,),
                                    data_base_path=meta_cfg.data_base_path,
                                )
                            )
                            if (
                                target_cfg.consis_dataloader_instance.name
                                == "TIF_txt_Dataset"
                            ):
                                consis_loader_cfg = target_cfg.consis_dataloader_instance.create_consis_config(
                                    perturbed_dir=(pred_dir_path,),
                                    unperturbed_dir=(none_pred_path,),
                                    data_base_path=meta_cfg.data_base_path,
                                )
                            else:
                                assert isinstance(
                                    target_cfg.consis_dataloader_instance,
                                    Eval_TIF_DataloaderMetaConfig,
                                )
                                consis_loader_cfg = target_cfg.consis_dataloader_instance.create_consis_config(
                                    perturbed_dir=(pred_dir_path,),
                                    unperturbed_dir=(none_pred_path,),
                                )

                    else:
                        assert_never(meta_cfg.segmentation_mode)

                    wandb_cfg = WandbConfig(
                        project=project_name,
                        name=f"{source_model.model_name}_{transfer_title_abbrev}_{save_name}",
                        mode="online",
                    )

                    if (a := augs_cfg[aug_name]) != None:
                        transforms = target_cfg.loader.transformer["raw"].insert(
                            -2, a.model_dump()
                        )
                        pred_loader = target_cfg.loader.model_copy(
                            update={"transformer": transforms}
                        )

                    else:
                        pred_loader = target_cfg.loader

                    if pred_loader.dataset in (
                        "StandardHDF5Dataset",
                        "S_BIAD1410_Dataset",
                    ):
                        pred_loader_cfg = pred_loader.create_config(
                            output_dir=pred_dir_path,
                            data_base_path=meta_cfg.data_base_path,
                        )

                    elif pred_loader.dataset in (
                        "Standard_TIF_Dataset",
                        "HeLaNuc_Dataset",
                        "Hoechst_Dataset",
                        "TIF_txt_Dataset",
                    ):
                        pred_loader_cfg = pred_loader.create_config(
                            output_dir=pred_dir_path,
                            data_base_path=meta_cfg.data_base_path,
                        )

                    else:
                        assert_never(pred_loader.dataset)

                    yaml_dir_path = "/".join(pred_dir_path.split("/")[:-1])
                    if target_cfg.filter_results is not None:
                        filter_patches_cfg = target_cfg.filter_results.create_config(
                            data_base_path=meta_cfg.data_base_path,
                        )
                    else:
                        filter_patches_cfg = None

                    if meta_cfg.run_mode == "full":
                        assert (
                            meta_cfg.eval_settings is not None
                        ), "Eval settings cannot be None for run mode {meta_cfg.run_mode}"
                        assert (
                            meta_cfg.consistency_settings is not None
                        ), "Consistency settings cannot be None for run mode {meta_cfg.run_mode}"
                        eval_metric_cfg = meta_cfg.eval_settings

                        consis_metric_cfg = meta_cfg.consistency_settings

                        eval_cfg = EvaluateConfig(
                            eval_dataloader=eval_loader_cfg,
                            eval_metric=eval_metric_cfg,
                        )
                        consis_cfg = ConsistencyConfig(
                            consistency_dataloader=consis_loader_cfg,
                            consistency_metric=consis_metric_cfg,
                        )

                        summary_results_cfg = SummaryResultsConfig(
                            filter_patches=filter_patches_cfg,
                            output_path=pred_dir_path,
                            eval_key=eval_cfg.eval_metric.eval_save_key,
                            consis_key=consis_cfg.consistency_metric.save_key,
                            overwrite_scores=meta_cfg.summary_results.overwrite_scores,
                            # save_select_patches=meta_cfg.save_results.save_select_patches,
                        )
                        yaml_save_path = Path(yaml_dir_path) / f"{save_name}.yml"

                        yaml_dict_order = [
                            {"wandb": wandb_cfg.model_dump()},
                            {"model_path": source_model_path},
                            {"summary_results": summary_results_cfg.model_dump()},
                            {"model": model_cfg.model_dump()},
                            {"predictor": predictor_cfg.model_dump()},
                            {"loaders": pred_loader_cfg.model_dump()},
                            {"evaluation": eval_cfg.model_dump()},
                            {"consistency": consis_cfg.model_dump()},
                        ]
                    elif meta_cfg.run_mode == "consistency":
                        assert (
                            meta_cfg.consistency_settings is not None
                        ), "Consistency settings cannot be None for run mode {meta_cfg.run_mode}"

                        consis_metric_cfg = meta_cfg.consistency_settings

                        consis_cfg = ConsistencyConfig(
                            consistency_dataloader=consis_loader_cfg,
                            consistency_metric=consis_metric_cfg,
                        )
                        summary_results_cfg = SummaryResultsConfig(
                            filter_patches=filter_patches_cfg,
                            output_path=pred_dir_path,
                            eval_key=None,
                            consis_key=consis_cfg.consistency_metric.save_key,
                            overwrite_scores=meta_cfg.summary_results.overwrite_scores,
                            # save_select_patches=meta_cfg.save_results.save_select_patches,
                        )
                        yaml_save_path = (
                            Path(yaml_dir_path)
                            / f"{save_name}_{consis_cfg.consistency_metric.save_key}.yml"
                        )
                        yaml_dict_order = [
                            {"summary_results": summary_results_cfg.model_dump()},
                            {"consistency": consis_cfg.model_dump()},
                        ]
                    elif meta_cfg.run_mode == "evaluation":
                        assert (
                            meta_cfg.eval_settings is not None
                        ), "Eval settings cannot be None for run mode {meta_cfg.run_mode}"
                        eval_metric_cfg = meta_cfg.eval_settings

                        eval_cfg = EvaluateConfig(
                            eval_dataloader=eval_loader_cfg,
                            eval_metric=eval_metric_cfg,
                        )

                        summary_results_cfg = SummaryResultsConfig(
                            filter_patches=filter_patches_cfg,
                            output_path=pred_dir_path,
                            eval_key=eval_cfg.eval_metric.eval_save_key,
                            consis_key=None,
                            overwrite_scores=meta_cfg.summary_results.overwrite_scores,
                            # save_select_patches=meta_cfg.summary_results.save_select_patches,
                        )

                        yaml_save_path = (
                            Path(yaml_dir_path)
                            / f"{save_name}_{eval_cfg.eval_metric.eval_save_key}_eval.yml"
                        )
                        yaml_dict_order = [
                            {"summary_results": summary_results_cfg.model_dump()},
                            {"evaluation": eval_cfg.model_dump()},
                        ]
                    else:
                        assert_never(meta_cfg.run_mode)

                    yaml_paths.setdefault(transfer_title, []).append(yaml_save_path)
                    save_yaml(
                        yaml_order=yaml_dict_order,
                        yaml_path=yaml_save_path,
                        overwrite=meta_cfg.overwrite_yaml,
                    )
            # for yaml_paths at key transfer_title if path contains "none" then ensure it is at index zero
            yaml_paths[transfer_title].sort(key=lambda x: 0 if "none" in str(x) else 1)
    return yaml_paths
