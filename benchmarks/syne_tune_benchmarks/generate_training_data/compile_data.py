import logging
import os
import random

from pathlib import Path
from argparse import ArgumentParser
from syne_tune.util import catchtime

from load_data import get_metadata, create_history_from_results

validation_benchmarks = ['fcnet-protein', 
                         'branin',
                         'imagenet_resnet_batch_size_512',
                        'tabrepo_CatBoost_2dplanes',
                        'tabrepo_CatBoost_APSFailure',
                        'tabrepo_CatBoost_Airlines_DepDelay_10M',
                        'tabrepo_CatBoost_Allstate_Claims_Severity',
                        'tabrepo_CatBoost_Amazon_employee_access',
                        'tabrepo_CatBoost_Australian',
                        'tabrepo_CatBoost_Bioresponse',
                        'tabrepo_CatBoost_Brazilian_houses',
                        'tabrepo_CatBoost_Buzzinsocialmedia_Twitter',
                        'tabrepo_CatBoost_CIFAR_10',
                        'tabrepo_CatBoost_Click_prediction_small',
                        'tabrepo_CatBoost_Devnagari-Script',
                        'tabrepo_CatBoost_Diabetes130US',
                        'tabrepo_CatBoost_Fashion-MNIST',
                        'tabrepo_CatBoost_GAMETES_Epistasis_2-Way_1000atts_0_4H_EDM-1_EDM-1_1',
                        'tabrepo_CatBoost_GAMETES_Epistasis_2-Way_20atts_0_1H_EDM-1_1',
                        'tabrepo_CatBoost_GAMETES_Epistasis_2-Way_20atts_0_4H_EDM-1_1',
                        'tabrepo_CatBoost_GAMETES_Epistasis_3-Way_20atts_0_2H_EDM-1_1',
                        'tabrepo_CatBoost_GAMETES_Heterogeneity_20atts_1600_Het_0_4_0_2_50_EDM-2_001',
                        'tabrepo_CatBoost_GAMETES_Heterogeneity_20atts_1600_Het_0_4_0_2_75_EDM-2_001',
                        'tabrepo_CatBoost_GTSRB-HOG01',
                        'tabrepo_CatBoost_GTSRB-HOG02',
                        'tabrepo_CatBoost_GTSRB-HOG03',
                        'tabrepo_CatBoost_GTSRB-HueHist',
                        'tabrepo_CatBoost_GesturePhaseSegmentationProcessed',
                        'tabrepo_CatBoost_Higgs',
                        'tabrepo_CatBoost_Indian_pines',
                        'tabrepo_CatBoost_Internet-Advertisements',
                        'tabrepo_CatBoost_KDDCup09-Upselling',
                        'tabrepo_CatBoost_KDDCup09_appetency',
                        'tabrepo_CatBoost_Kuzushiji-MNIST',
                        'tabrepo_CatBoost_LED-display-domain-7digit',
                        'tabrepo_CatBoost_MIP-2016-regression',
                        'tabrepo_CatBoost_MagicTelescope',
                        'tabrepo_CatBoost_Mercedes_Benz_Greener_Manufacturing',
                        'tabrepo_CatBoost_MiceProtein',
                        'tabrepo_CatBoost_MiniBooNE',
                        'tabrepo_CatBoost_Moneyball',
                        'tabrepo_CatBoost_OVA_Breast',
                        'tabrepo_CatBoost_OVA_Colon',
                        'tabrepo_CatBoost_OVA_Endometrium',
                        'tabrepo_CatBoost_OVA_Kidney',
                        'tabrepo_CatBoost_OVA_Lung',
                        'tabrepo_CatBoost_OVA_Ovary',
                        'tabrepo_CatBoost_OVA_Prostate',
                        'tabrepo_CatBoost_OnlineNewsPopularity',
                        'tabrepo_CatBoost_PhishingWebsites',
                        'tabrepo_CatBoost_QSAR-TID-10980',
                        'tabrepo_CatBoost_QSAR-TID-11',
                        'tabrepo_CatBoost_Run_or_walk_information',
                        'tabrepo_CatBoost_SAT11-HAND-runtime-regression',
                        'tabrepo_CatBoost_Santander_transaction_value',
                        'tabrepo_CatBoost_Satellite',
                        'tabrepo_CatBoost_SpeedDating',
                        'tabrepo_CatBoost_Titanic',
                        'tabrepo_CatBoost_Traffic_violations',
                        'tabrepo_CatBoost_UMIST_Faces_Cropped',
                        'tabrepo_CatBoost_Yolanda',
                        'tabrepo_CatBoost_abalone',
                        'tabrepo_CatBoost_ada',
                        'tabrepo_CatBoost_adult',
                        'tabrepo_CatBoost_ailerons',
                        'tabrepo_CatBoost_airlines',
                        'tabrepo_CatBoost_albert',
                        'tabrepo_CatBoost_analcatdata_authorship',
                        'tabrepo_CatBoost_analcatdata_dmft',
                        'tabrepo_CatBoost_anneal',
                        'tabrepo_CatBoost_arcene',
                        'tabrepo_CatBoost_arsenic-female-bladder',
                        'tabrepo_CatBoost_artificial-characters',
                        'tabrepo_CatBoost_autoUniv-au1-1000',
                        'tabrepo_CatBoost_autoUniv-au6-750',
                        'tabrepo_CatBoost_autoUniv-au7-1100',
                        'tabrepo_CatBoost_autoUniv-au7-700',
                        'tabrepo_CatBoost_balance-scale',
                        'tabrepo_CatBoost_bank-marketing',
                        'tabrepo_CatBoost_bank32nh',
                        'tabrepo_CatBoost_bank8FM',
                        'tabrepo_CatBoost_baseball',
                        'tabrepo_CatBoost_black_friday',
                        'tabrepo_CatBoost_blood-transfusion-service-center',
                        'tabrepo_CatBoost_boston',
                        'tabrepo_CatBoost_boston_corrected',
                        'tabrepo_CatBoost_car',
                        'tabrepo_CatBoost_cardiotocography',
                        'tabrepo_CatBoost_christine',
                        'tabrepo_CatBoost_churn',
                        'tabrepo_CatBoost_climate-model-simulation-crashes',
                        'tabrepo_CatBoost_cmc',
                        'tabrepo_CatBoost_cnae-9',
                        'tabrepo_CatBoost_colleges',
                        'tabrepo_CatBoost_colleges_usnews',
                        'tabrepo_CatBoost_collins',
                        'tabrepo_CatBoost_connect-4',
                        'tabrepo_CatBoost_covertype',
                        'tabrepo_CatBoost_cpu_act',
                        'tabrepo_CatBoost_cpu_small',
                        'tabrepo_CatBoost_credit-g',
                        'tabrepo_CatBoost_cylinder-bands',
                        'tabrepo_CatBoost_delta_ailerons',
                        'tabrepo_CatBoost_delta_elevators',
                        'tabrepo_CatBoost_diabetes',
                        'tabrepo_CatBoost_diamonds',
                        'tabrepo_CatBoost_dilbert',
                        'tabrepo_CatBoost_dna',
                        'tabrepo_CatBoost_dresses-sales',
                        'tabrepo_CatBoost_eating',
                        'tabrepo_CatBoost_eeg-eye-state',
                        'tabrepo_CatBoost_electricity',
                        'tabrepo_CatBoost_elevators',
                        'tabrepo_CatBoost_eucalyptus',
                        'tabrepo_CatBoost_eye_movements',
                        'tabrepo_CatBoost_fabert',
                        'tabrepo_CatBoost_fars',
                        'tabrepo_CatBoost_first-order-theorem-proving',
                        'tabrepo_CatBoost_fri_c0_1000_5',
                        'tabrepo_CatBoost_fri_c0_500_5',
                        'tabrepo_CatBoost_fri_c1_1000_50',
                        'tabrepo_CatBoost_fri_c2_1000_25',
                        'tabrepo_CatBoost_fri_c2_500_50',
                        'tabrepo_CatBoost_fri_c3_1000_10',
                        'tabrepo_CatBoost_fri_c3_1000_25',
                        'tabrepo_CatBoost_fri_c3_500_10',
                        'tabrepo_CatBoost_fri_c3_500_50',
                        'tabrepo_CatBoost_fri_c4_500_100',
                        'tabrepo_CatBoost_fried',
                        'tabrepo_CatBoost_gina',
                        'tabrepo_CatBoost_guillermo',
                        'tabrepo_CatBoost_har',
                        'tabrepo_CatBoost_helena',
                        'tabrepo_CatBoost_hill-valley',
                        'tabrepo_CatBoost_hiva_agnostic',
                        'tabrepo_CatBoost_house_16H',
                        'tabrepo_CatBoost_house_prices_nominal',
                        'tabrepo_CatBoost_house_sales',
                        'tabrepo_CatBoost_houses',
                        'tabrepo_CatBoost_hypothyroid',
                        'tabrepo_CatBoost_ilpd',
                        'tabrepo_CatBoost_isolet',
                        'tabrepo_CatBoost_jannis',
                        'tabrepo_CatBoost_jasmine',
                        'tabrepo_CatBoost_jm1',
                        'tabrepo_CatBoost_jungle_chess_2pcs_raw_endgame_complete',
                        'tabrepo_CatBoost_kc1',
                        'tabrepo_CatBoost_kc2',
                        'tabrepo_CatBoost_kdd_el_nino-small',
                        'tabrepo_CatBoost_kdd_internet_usage',
                        'tabrepo_CatBoost_kick',
                        'tabrepo_CatBoost_kin8nm',
                        'tabrepo_CatBoost_kr-vs-k',
                        'tabrepo_CatBoost_kropt',
                        'tabrepo_CatBoost_ldpa',
                        'tabrepo_CatBoost_led24',
                        'tabrepo_CatBoost_letter',
                        'tabrepo_CatBoost_madeline',
                        'tabrepo_CatBoost_madelon',
                        'tabrepo_CatBoost_mammography',
                        'tabrepo_CatBoost_mc1',
                        'tabrepo_CatBoost_meta',
                        'tabrepo_CatBoost_mfeat-factors',
                        'tabrepo_CatBoost_micro-mass',
                        'tabrepo_CatBoost_microaggregation2',
                        'tabrepo_CatBoost_mnist_784',
                        'tabrepo_CatBoost_mozilla4',
                        'tabrepo_CatBoost_no2',
                        'tabrepo_CatBoost_nomao',
                        'tabrepo_CatBoost_numerai28_6',
                        'tabrepo_CatBoost_nursery',
                        'tabrepo_CatBoost_nyc-taxi-green-dec-2016',
                        'tabrepo_CatBoost_okcupid-stem',
                        'tabrepo_CatBoost_one-hundred-plants-margin',
                        'tabrepo_CatBoost_optdigits',
                        'tabrepo_CatBoost_ozone-level-8hr',
                        'tabrepo_CatBoost_page-blocks',
                        'tabrepo_CatBoost_parity5_plus_5',
                        'tabrepo_CatBoost_pbcseq',
                        'tabrepo_CatBoost_pc1',
                        'tabrepo_CatBoost_pc2',
                        'tabrepo_CatBoost_pc3',
                        'tabrepo_CatBoost_pc4',
                        'tabrepo_CatBoost_pendigits',
                        'tabrepo_CatBoost_philippine',
                        'tabrepo_CatBoost_phoneme',
                        'tabrepo_CatBoost_pm10',
                        'tabrepo_CatBoost_pokerhand',
                        'tabrepo_CatBoost_pol',
                        'tabrepo_CatBoost_pollen',
                        'tabrepo_CatBoost_porto-seguro',
                        'tabrepo_CatBoost_puma32H',
                        'tabrepo_CatBoost_puma8NH',
                        'tabrepo_CatBoost_qsar-biodeg',
                        'tabrepo_CatBoost_quake',
                        'tabrepo_CatBoost_riccardo',
                        'tabrepo_CatBoost_ringnorm',
                        'tabrepo_CatBoost_rmftsa_ladata',
                        'tabrepo_CatBoost_robert',
                        'tabrepo_CatBoost_satimage',
                        'tabrepo_CatBoost_segment',
                        'tabrepo_CatBoost_semeion',
                        'tabrepo_CatBoost_sensory',
                        'tabrepo_CatBoost_sf-police-incidents',
                        'tabrepo_CatBoost_shuttle',
                        'tabrepo_CatBoost_socmob',
                        'tabrepo_CatBoost_soybean',
                        'tabrepo_CatBoost_space_ga',
                        'tabrepo_CatBoost_spambase',
                        'tabrepo_CatBoost_splice',
                        'tabrepo_CatBoost_spoken-arabic-digit',
                        'tabrepo_CatBoost_steel-plates-fault',
                        'tabrepo_CatBoost_sylvine',
                        'tabrepo_CatBoost_synthetic_control',
                        'tabrepo_CatBoost_tamilnadu-electricity',
                        'tabrepo_CatBoost_tecator',
                        'tabrepo_CatBoost_texture',
                        'tabrepo_CatBoost_tokyo1',
                        'tabrepo_CatBoost_topo_2_1',
                        'tabrepo_CatBoost_twonorm',
                        'tabrepo_CatBoost_us_crime',
                        'tabrepo_CatBoost_vehicle',
                        'tabrepo_CatBoost_visualizing_soil',
                        'tabrepo_CatBoost_volcanoes-a2',
                        'tabrepo_CatBoost_volcanoes-a3',
                        'tabrepo_CatBoost_volcanoes-a4',
                        'tabrepo_CatBoost_volcanoes-b1',
                        'tabrepo_CatBoost_volcanoes-b2',
                        'tabrepo_CatBoost_volcanoes-b5',
                        'tabrepo_CatBoost_volcanoes-b6',
                        'tabrepo_CatBoost_volcanoes-d1',
                        'tabrepo_CatBoost_volcanoes-d4',
                        'tabrepo_CatBoost_volcanoes-e1',
                        'tabrepo_CatBoost_volkert',
                        'tabrepo_CatBoost_walking-activity',
                        'tabrepo_CatBoost_wall-robot-navigation',
                        'tabrepo_CatBoost_waveform-5000',
                        'tabrepo_CatBoost_wilt',
                        'tabrepo_CatBoost_wind',
                        'tabrepo_CatBoost_wine-quality-red',
                        'tabrepo_CatBoost_wine-quality-white',
                        'tabrepo_CatBoost_wine_quality',
                        'tabrepo_CatBoost_yeast',
                        'tabrepo_CatBoost_yprop_4_1',
                         ]

if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)

    parser = ArgumentParser()
    parser.add_argument(
        "--path",
        type=str,
        required=True,
        help="path where to find the results",
    )
    parser.add_argument(
        "--max_seed",
        type=int,
        required=False,
        default=30,
    )
    parser.add_argument(
        "--num_permutation",
        type=int,
        required=False,
        default=5,
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="path to store the results",
    )
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.8,
        help="ratio of data used for training",
    )

    methods = [
        "REA",
        "TPE",
        "BORE",
        "CQR",
        "RS",
        "HEBO"
    ]

    args, _ = parser.parse_known_args()

    print(args.__dict__)
    assert Path(args.path).exists()
    max_seed = args.max_seed
    max_num_trials = 100

    path = Path(args.path)
    output_path = Path(args.output_path)
    os.makedirs(output_path, exist_ok=True)
    experiment_filter = None

    with catchtime("load benchmark results"):

        with catchtime("Load metadata"):
            metadatas = get_metadata(root=path)

        # todo strict metadata filtering as the one above may fail
        methods = set(methods) if methods is not None else None
        metadatas = {
            k: v
            for k, v in metadatas.items()
            if (max_seed is None or v["seed"] < max_seed)
               and (methods is None or v["algorithm"] in methods)
        }
        if experiment_filter:
            metadatas = {k: v for k, v in metadatas.items() if experiment_filter(v)}
        print(f"loaded {len(metadatas)} experiment metadata")
        # metadatas = {k: v for k, v in metadatas.items() if "yahpo" not in v["benchmark"]}

        with catchtime("Load results dataframes"):
            # load results in parallel

            hist_train = []
            hist_valid = []
            for name, metadata in metadatas.items():
#                try:
                    benchmark_name = metadata['benchmark']
                    if benchmark_name in validation_benchmarks:
                        hist_valid.extend(create_history_from_results(name, metadata, path, max_num_trials, args.num_permutation))
                    else:
                        hist_train.extend(create_history_from_results(name, metadata, path, max_num_trials, args.num_permutation))
#                except Exception as e:
#                    print(f"Error processing {name}: {e}")
#                    continue
            #        hist = parfor(
            #            lambda name, metadata: create_history_from_results(name, metadata, path),
            #            inputs=list(metadatas.items()),
            #            engine=engine,
            #        )
            random.shuffle(hist_train)
            for split in ['train', 'valid']:
                file_name = f"{split}.txt"
                if split == 'train':
                    hist_split = hist_train
                else:
                    hist_split = hist_valid
                with open(str(output_path / file_name), 'w', encoding='utf-8') as f:
                    f.write('\n'.join(hist_split))