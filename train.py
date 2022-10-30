import argparse
import os
import utils

if __name__ == "__main__":
    parse = argparse.ArgumentParser()
    parse.add_argument("--data_path", type=str, required=True)
    parse.add_argument("--sample_rate", type=int)
    parse.add_argument("--window_size", type=int)
    parse.add_argument("--n_fft", type=int)
    parse.add_argument("--hop_length", type=int)
    parse.add_argument("--n_mels", type=int)
    parse.set_defaults(
        sample_rate=44100,
        window_size=256,
        n_fft=2048,
        n_mels=40
    )
    args = parse.parse_args()

    _valid_sample_rates = [8000, 16000, 24000, 44000, 44100]
    assert os.path.exists(args.data_path) and os.path.exists(os.path.join(args.data_path, "evaluation_setup")) \
           and os.path.exists(os.path.join(args.data_path, "audio")) and \
           os.path.exists(os.path.join(args.data_path, "meta")), "Please use a valid data path"
    assert args.window_size > 0, "Please use a valid value for window size"
    assert args.n_mels > 0, "Please use a valid value for filter mels"
    assert args.n_fft > 0, "Please use a valid value for n_fft"
    assert args.sample_rate in _valid_sample_rates, f"Please use a valid value for sample rate ({_valid_sample_rates})"
    assert args.hop_length is None or args.hop_length > 0, "Please use a valid value for hop_length"

    if args.hop_length is None:
        args.hop_length = args.n_fft // 2

    _folds = [1, 2, 3, 4]

    for fold in _folds:
        print("\n--------------------------------")
        print(f"FOLD {fold}")
        print("--------------------------------\n")

        _files_path = os.path.join(args.data_path, "evaluation_setup")
        _train_path = os.path.join(_files_path, f"street_fold{fold}_train.txt")
        _evaluate_path = os.path.join(_files_path, f"street_fold{fold}_evaluate.txt")
        _test_path = os.path.join(_files_path, f"street_fold{fold}_test.txt")
        _meta_folder_path = os.path.join(args.data_path, "meta")
                                   
        train_df = utils.open_file(path=_train_path,
                                   meta_path=_meta_folder_path)
        
        train_features, encoded_matrix_train = utils.extract_features(path=args.data_path,
                                                                      sample_rate=args.sample_rate,
                                                                      n_fft=args.n_fft,
                                                                      hop_length=args.hop_length,
                                                                      n_mels=args.n_mels,
                                                                      df=train_df,
                                                                      window_size=args.window_size)
        
        print(train_features.shape, encoded_matrix_train.shape)
        
        evaluate_df = utils.open_file(path=_evaluate_path,
                                      meta_path=_meta_folder_path)
        
        evaluate_features, encoded_matrix_evaluate = utils.extract_features(path=args.data_path,
                                                                            sample_rate=args.sample_rate,
                                                                            n_fft=args.n_fft,
                                                                            hop_length=args.hop_length,
                                                                            n_mels=args.n_mels,
                                                                            df=evaluate_df,
                                                                            window_size=args.window_size)
        
        print(evaluate_features.shape, encoded_matrix_evaluate.shape)
        
        test_df = utils.open_file(path=_test_path,
                                  meta_path=_meta_folder_path,
                                  is_test=True)
        
        test_features, encoded_matrix_test = utils.extract_features(path=args.data_path,
                                                                    sample_rate=args.sample_rate,
                                                                    n_fft=args.n_fft,
                                                                    hop_length=args.hop_length,
                                                                    n_mels=args.n_mels,
                                                                    df=test_df,
                                                                    window_size=args.window_size)
        
        print(test_features.shape, encoded_matrix_test.shape)