"""Explicit CLI stages: inspect metadata, bounded fetch, ingest, split, check."""
import argparse
import json
import os
from pathlib import Path

from .config import PipelineConfig
from .dataset import check_hdf5, create_user_split, ingest_alyx
from .sources import SOURCE_REVISION, discover_alyx, fetch_alyx, reseal_legacy_receipt


def main(argv=None):
    parser = argparse.ArgumentParser(prog="abc-iueh")
    commands = parser.add_subparsers(dest="command", required=True)
    discover = commands.add_parser("discover", help="Fetch pinned public DVC metadata, not recordings")
    discover.add_argument("--output", default="data/alyx_inventory.json")
    discover.add_argument("--revision", default=SOURCE_REVISION)
    fetch = commands.add_parser("fetch", help="Bounded WhoIsAlyx motion-only download")
    fetch.add_argument("--manifest", default="data/alyx_inventory.json")
    fetch.add_argument("--data-root", default="data")
    fetch.add_argument("--noncommercial", action="store_true")
    fetch.add_argument("--max-files", type=int, default=2)
    fetch.add_argument("--max-total-mib", type=int, default=256)
    fetch.add_argument("--users", nargs="+")
    fetch.add_argument("--receipt", default="data/download_receipt_v2.json",
                       help="new v2 receipt path; existing receipts are never overwritten")
    fetch.add_argument("--previous-receipt", default=None,
                       help="optional earlier receipt to extend; defaults to the legacy data/download_receipt.json when present")
    ingest = commands.add_parser("ingest", help="Verify recordings and build gap-safe HDF5")
    ingest.add_argument("--data-root", default="data")
    ingest.add_argument("--output", default="data/processed/alyx.h5")
    ingest.add_argument("--report", default="data/quality_report.json")
    ingest.add_argument("--config", default="configs/pipeline.json")
    ingest.add_argument("--manifest", default="data/alyx_inventory.json",
                        help="complete authenticated source inventory used for this receipt")
    ingest.add_argument("--receipt", default="data/download_receipt_v2.json",
                        help="versioned receipt to validate and ingest")
    reseal = commands.add_parser("reseal-receipt", help="Metadata-only validate and version a legacy download receipt")
    reseal.add_argument("--manifest", default="data/alyx_inventory.json")
    reseal.add_argument("--receipt", default="data/download_receipt.json")
    reseal.add_argument("--data-root", default="data")
    reseal.add_argument("--output", required=True,
                        help="new output path; historical receipt and inventory are never overwritten")
    split = commands.add_parser("split", help="Lock disjoint user assignments before windowing")
    split.add_argument("--hdf5", default="data/processed/alyx.h5")
    split.add_argument("--output", default="data/user_split.json")
    split.add_argument("--config", default="configs/pipeline.json")
    split.add_argument("--smoke", action="store_true", help="Non-training manifest when cohort is too small")
    check = commands.add_parser("check", help="Validate schema/clock/units without forecast evaluation")
    check.add_argument("--hdf5", default="data/processed/alyx.h5")
    check.add_argument("--config", default="configs/pipeline.json")
    stage_commands = {
        "features": "Build and seal train/development causal features; final is forbidden",
        "seal-features": "Validate and seal an existing feature receipt without regenerating artifacts",
        "baselines": "Run analytic baselines on a verified train/development feature bundle",
        "neural-baseline": "Fit the frozen neural reference on train; evaluate train/development only",
    }
    for name, help_text in stage_commands.items():
        command = commands.add_parser(name, help=help_text)
        command.add_argument("--hdf5", default="data/processed/alyx.h5")
        command.add_argument("--split", default="data/user_split.json")
        command.add_argument("--config", default="configs/baseline_features.json")
        command.add_argument("--output-dir", "--run-dir", dest="output_dir", default="data/baseline_runs/20261028_r80")
    selectors = commands.add_parser("selectors", help="Run the frozen matched selector study on local CUDA; final is forbidden")
    selectors.add_argument("--hdf5", default="data/processed/alyx.h5")
    selectors.add_argument("--split", default="data/user_split.json")
    selectors.add_argument("--baseline-dir", default="data/baseline_runs/20261028_r80")
    selectors.add_argument("--output-dir", default="data/selector_runs/20261028_r80_gpu")
    selectors.add_argument("--config", default="configs/selector_study.json")
    selectors.add_argument("--device", default="cuda")
    sensitivity = commands.add_parser("selector-sensitivity", help="Run separately bound R128/R256 h1/h20 CUDA ablations")
    sensitivity.add_argument("--hdf5", default="data/processed/alyx.h5")
    sensitivity.add_argument("--split", default="data/user_split.json")
    sensitivity.add_argument("--baseline-dir", default="data/baseline_runs/20261028_r80")
    sensitivity.add_argument("--main-study-dir", default="data/selector_runs/20261028_r80_gpu")
    sensitivity.add_argument("--output-dir", default="data/selector_runs/20261028_R_sensitivity_gpu")
    sensitivity.add_argument("--config", default="configs/selector_sensitivity.json")
    sensitivity.add_argument("--device", default="cuda")
    inference = commands.add_parser("inference-audit", help="Audit private full graphs on a fixed train/dev sample; final is forbidden")
    inference.add_argument("--hdf5", default="data/processed/alyx.h5")
    inference.add_argument("--split", default="data/user_split.json")
    inference.add_argument("--baseline-dir", default="data/baseline_runs/20261028_r80")
    inference.add_argument("--main-study-dir", default="data/selector_runs/20261028_r80_gpu")
    inference.add_argument("--output-dir", default="data/inference_runs/20261028_fp32_audit")
    inference.add_argument("--config", default="configs/inference_audit.json")
    args = parser.parse_args(argv)
    try:
        if args.command == "discover":
            result = discover_alyx(args.output, args.revision)
            result = {k: v for k, v in result.items() if k != "records"}
        elif args.command == "fetch":
            records = fetch_alyx(args.manifest, args.data_root, noncommercial=args.noncommercial,
                                 max_files=args.max_files, max_total_bytes=args.max_total_mib * 1024**2,
                                 users=args.users, receipt_path=args.receipt,
                                 previous_receipt_path=args.previous_receipt)
            result = {"verified_files": len(records), "verified_bytes": sum(r["expected_bytes"] for r in records),
                      "receipt_path": str(args.receipt)}
        elif args.command == "ingest":
            result = ingest_alyx(args.data_root, args.output, args.report, PipelineConfig.load(args.config),
                                 inventory_path=args.manifest, receipt_path=args.receipt)
            result = {k: v for k, v in result.items() if k not in {"sessions", "eligible_users"}}
        elif args.command == "reseal-receipt":
            result = reseal_legacy_receipt(args.manifest, args.receipt, args.output, args.data_root)
        elif args.command == "split":
            result = create_user_split(args.hdf5, args.output, PipelineConfig.load(args.config), smoke=args.smoke)
            result = {**{k: v for k, v in result.items() if k != "partitions"},
                      "partition_counts": {k: len(v) for k, v in result["partitions"].items()}}
        elif args.command == "inference-audit":
            os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
            from .inference_audit import run_inference_audit, validate_inference_config
            from .stage import LockedDevelopmentStage
            config = validate_inference_config(json.loads(Path(args.config).read_text(encoding="utf-8")))
            result = run_inference_audit(LockedDevelopmentStage(args.hdf5, args.split), args.baseline_dir,
                                         args.main_study_dir, args.output_dir, config, config_path=args.config)
        elif args.command == "selector-sensitivity":
            os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
            from .selector_sensitivity import run_sensitivity, validate_sensitivity_config
            from .stage import LockedDevelopmentStage
            config = validate_sensitivity_config(json.loads(Path(args.config).read_text(encoding="utf-8")))
            result = run_sensitivity(LockedDevelopmentStage(args.hdf5, args.split), args.baseline_dir,
                                     args.main_study_dir, args.output_dir, config, device=args.device,
                                     config_path=args.config)
        elif args.command == "selectors":
            # Must precede importing Torch or creating a CUDA context.
            os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
            from .selector_experiment import run_selector_study, validate_selector_config
            from .stage import LockedDevelopmentStage
            config = validate_selector_config(json.loads(Path(args.config).read_text(encoding="utf-8")))
            stage = LockedDevelopmentStage(args.hdf5, args.split)
            result = run_selector_study(stage, args.baseline_dir, args.output_dir, config,
                                        device=args.device, config_path=args.config)
        elif args.command in stage_commands:
            from .stage import (LockedDevelopmentStage, build_feature_cache, seal_feature_receipt,
                                validate_experiment_config, validate_feature_bundle)
            config = validate_experiment_config(json.loads(Path(args.config).read_text(encoding="utf-8")))
            stage = LockedDevelopmentStage(args.hdf5, args.split)
            directory = Path(args.output_dir)
            if args.command == "features":
                result = build_feature_cache(stage, directory / "features.h5", directory / "scalers.json",
                                             evidence_length=config["evidence_length"],
                                             include_complete_horizon=config["complete_horizon"],
                                             config_path=args.config)
            elif args.command == "seal-features":
                result = seal_feature_receipt(stage, directory, config_path=args.config)
            else:
                validate_feature_bundle(stage, directory, config_path=args.config)
                if args.command == "baselines":
                    from .experiments import run_analytic
                    result = run_analytic(stage, directory, config, config_path=args.config)
                else:
                    from .neural_experiment import run_neural
                    result, _ = run_neural(stage, directory / "features.h5", directory, config,
                                           config_path=args.config)
        else:
            result = check_hdf5(args.hdf5, PipelineConfig.load(args.config))
        print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))
        return 0
    except (ValueError, PermissionError, FileNotFoundError, FileExistsError) as error:
        parser.exit(2, f"error: {error}\n")


if __name__ == "__main__":
    main()
