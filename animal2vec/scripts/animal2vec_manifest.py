#!/usr/bin/env python3 -u
# Copyright (c) Max Planck Institute of Animal Behavior
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Data pre-processing: build vocabularies and binarize training data.
"""

import argparse
import os
import re
from pathlib import Path
from typing import Iterator, List, Tuple

import h5py
import numpy as np
import pandas as pd
import soundfile
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
from numpy.typing import NDArray

# local packages
# from utils import get_files


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "root", metavar="DIR", help="root directory containing files to index"
    )
    parser.add_argument(
        "--valid-percent",
        default=0.2,
        type=float,
        metavar="D",
        help="percentage of data to use as validation set (between 0 and 1)",
    )
    parser.add_argument(
        "--n-split",
        default=5,
        type=int,
        help="How many splits should we do for cross validation",
    )
    parser.add_argument(
        "--dest", default=".", type=str, metavar="DIR", help="output directory"
    )
    parser.add_argument(
        "--ext", default="wav", type=str, metavar="EXT", help="extension to look for"
    )
    parser.add_argument(
        "--seed", default=1612, type=int, metavar="N", help="random seed"
    )
    parser.add_argument(
        "--few-shot",
        default=False,
        type=bool,
        help="Should we export a few-shot set of manifest files."
        "This will create for every train split 5 additional stratified "
        "few-shot train splits with percentages: .01, .1, .25, .5, .75",
    )
    parser.add_argument(
        "--leave-p-out",
        default=False,
        type=bool,
        help="Should we export a set of manifest files that follow the leave-p-out "
        "test strategy. This will randomly select p files that will never "
        "be used in pretrain or finetune-train, but only for finetune-eval."
        "p is chosen such that the test/train split is roughly 20/80."
        "For this testing strategy no few-shot variant will be exported, "
        "and no k-fold cross validation as well.",
    )
    parser.add_argument(
        "--path-must-contain",
        default=None,
        type=str,
        metavar="FRAG",
        help="if set, path must contain this substring for a file to be included in the manifest",
    )
    parser.add_argument(
        "--valid-set-individuals",
        type=str,
        default="",
        nargs="+",
        help="List of individual IDs that should be exclusively used for the validation set for the fine-tuning phase."
        "This is useful to test generalization to unseen individuals."
        "Will be ignored when leave-p-out is set.",
    )
    parser.add_argument(
        "--id-lookup-file-path",
        type=str,
        default="",
        help="File path that maps randomised file names to their non-randomised name which include individual information."
        "Will be used only when valid-set-individuals is set.",
    )

    return parser


def get_files(dir, re_obj=None):
    r"""
    Traverse a directory's subtree picking up all files of correct type
    Files can be selectively filtered by using a regular expression (re) object.
    :param dir: target directory
    :param re_obj: regexp (re) object.
    :return:  List of file paths (dir is prepended)

    Example, pick up files that end in .wav or .flac in a case independent manner:
      import re
      audio_re = re.compile(r".*\.(flac|wav)$", re.IGNORECASE)
      audio_files = get_files(some_dir, re_obj=audio_re)
    """

    files = []

    # Standard traversal with os.walk, see library docs
    for dirpath, dirnames, filenames in os.walk(dir):
        if re_obj is not None:
            # Only retain files that match the pattern
            filenames = [f for f in filenames if re_obj.match(f)]
        # Add full path of each filename to list of files
        for filename in filenames:
            files.append(os.path.join(dirpath, filename))

    return files


def flatten(l):
    """helper function for flattening a list of lists"""
    return [item for sublist in l for item in sublist]


def get_files_indices_one_individual(
    individual: str, labels_X: List[str], id_lookup_file_path: Path = ""
) -> List[int]:
    """
    Gets the indices of all files that belong to one individual
    """
    if not id_lookup_file_path:
        indices_indiv = [
            i
            for i in range(len(labels_X))
            if individual.lower() in Path(labels_X[i]).stem.lower()
        ]
        print(
            f"Found {len(indices_indiv)} files for individual {individual} based on file name matching."
        )
    else:
        lookup_file_df = pd.DataFrame(pd.read_csv(id_lookup_file_path, sep="\t"))
        lookup_file_df_one_indiv = lookup_file_df[lookup_file_df['OriginalName'].str.contains(f"_{individual}")]
        randomised_file_names = lookup_file_df_one_indiv['RandomizedName'].tolist()
        indices_indiv = []
        for file_name in randomised_file_names:
            idx_file_name = [i for i in range(len(labels_X)) if file_name in labels_X[i]]
            assert len(idx_file_name) <= 1
            if idx_file_name:
                indices_indiv.append(idx_file_name[0])

    return indices_indiv


def split_train_valid_by_individuals(
    labels_X: List[str], valid_set_individuals: List[str], id_lookup_file_path: Path
) -> Iterator[Tuple[NDArray, NDArray]]:
    """
    Splits the labelled dataset into train and valid set based on specific individuals
    """
    valid_set_indices = []
    for individual in valid_set_individuals:
        idx_ind_valid = get_files_indices_one_individual(
            individual, labels_X, id_lookup_file_path
        )
        valid_set_indices.extend(idx_ind_valid)

    train_set_indices = list(set(range(len(labels_X))) - set(valid_set_indices))
    it_ = iter([(np.array(train_set_indices), np.array(valid_set_indices))])
    return it_


def main(args):
    assert args.valid_percent >= 0 and args.valid_percent <= 1.0

    if not os.path.exists(args.dest):
        os.makedirs(args.dest)

    print(args)
    dir_path = os.path.realpath(args.root)
    # Get list of audio files
    ext_re = re.compile(r".*\.{}$".format(args.ext), re.IGNORECASE)
    audio_files = get_files(dir_path, ext_re)
    if len(audio_files) < 1:
        raise RuntimeError("No audio files were found.")

    # We have to iterate two times through all files to collect
    # necessary stats for stratified train / valid splits
    labels_X = []
    labels_y = []
    files_without_labels = []
    # class_counts_read = {}
    for fname in audio_files:  # First Pass
        file_path = os.path.realpath(fname)

        if args.path_must_contain and args.path_must_contain not in file_path:
            continue

        # We assume that audio files are in a wav subdirectory and that labels are in a lbl
        # subdirectory with npz extensions
        label_file = file_path.replace(os.sep + "wav" + os.sep, os.sep + "lbl" + os.sep)
        label_file = label_file.replace(".wav", ".h5")
        label_file_check = os.path.isfile(label_file)
        if label_file_check:
            with h5py.File(label_file, "r") as f:
                categorical_label = list(f["lbl_cat"])
            if (
                len(categorical_label) == 0
            ):  # if file has no labels, but labels generally exist
                files_without_labels.append(file_path)
            else:
                # if multiple labels are present in one file we
                # only consider the one that happened the least so far
                cl_unique, cl_counts = np.unique(categorical_label, return_counts=True)
                labels_X.append(file_path)
                labels_y.append(cl_unique)
        else:  # if no labels exist at all
            files_without_labels.append(file_path)

    if len(labels_y) > 0:
        # Report class labels/counts and verify examples for all classes
        unique_target_classes, unique_target_class_counts = np.unique(
            flatten(labels_y), return_counts=True
        )
        print("Classes/counts:  ", end=None)
        for class_id, class_count in zip(
            unique_target_classes, unique_target_class_counts
        ):
            print(f"{class_id}/{class_count} ", end="")
        # TODO when not all classes are in the input files it throws an error
        # TODO count is wrong when multiple signals of the same class are in the file, then, only one is counted
        # assert len(unique_target_classes) - 1 == max(unique_target_classes), \
        #     f"{max(unique_target_classes)} classes exist, " \
        #     f"only {len(unique_target_classes)} classes with examples:"

        targets = []
        for ll in labels_y:
            tmp_zero_target = np.zeros(max(unique_target_classes) + 1)
            tmp_zero_target[ll] = 1
            targets.append(tmp_zero_target)

    if args.leave_p_out:
        root_len = len(dir_path)
        # get the unique original file names
        unique_basenames = np.unique(
            [[os.path.basename(x[1 + root_len :])[:-18] for x in labels_X]]
        ).tolist()
        # select p files for the leave-p-out test strategy
        p = round(0.2 * len(unique_basenames))
        print("We are exporting leave-p-out manifest files with p={}".format(p))
        lof = np.random.choice(unique_basenames, p).tolist()  # leave-out-files
        test_index_lof = set(
            np.argwhere([any([y in x for y in lof]) for x in labels_X]).flatten()
        )
        train_index_lof = (
            set(np.arange(len(labels_X)).flatten()) - test_index_lof
        )  # calculate the complement
        files_without_labels_index_lof = (
            np.argwhere([not any([y in x for y in lof]) for x in files_without_labels])
            .squeeze()
            .tolist()
        )

        valid_f_lof = open(os.path.join(args.dest, "valid_lof.tsv"), "w")
        pretrain_f_lof = open(os.path.join(args.dest, "pretrain_lof.tsv"), "w")
        train_f_lof = open(os.path.join(args.dest, "train_lof.tsv"), "w")
        print(dir_path, file=valid_f_lof)  # header
        print(dir_path, file=pretrain_f_lof)  # header
        print(dir_path, file=train_f_lof)  # header

        for (
            filename_index
        ) in train_index_lof:  # Second Pass for train files with labels
            file_path = os.path.realpath(labels_X[filename_index])
            try:
                frames = soundfile.info(file_path).frames
            except soundfile.LibsndfileError as sE:
                print(sE)
                continue
            line = "{}\t{}".format(os.path.relpath(file_path, dir_path), frames)
            print(line, file=train_f_lof)
            print(line, file=pretrain_f_lof)
        for filename_index in test_index_lof:  # Second Pass for test files with labels
            file_path = os.path.realpath(labels_X[filename_index])
            try:
                frames = soundfile.info(file_path).frames
            except soundfile.LibsndfileError as sE:
                print(sE)
                continue
            line = "{}\t{}".format(os.path.relpath(file_path, dir_path), frames)
            print(line, file=valid_f_lof)
        for (
            filename_index
        ) in files_without_labels_index_lof:  # Second Pass for files without labels
            file_path = os.path.realpath(files_without_labels[filename_index])
            try:
                frames = soundfile.info(file_path).frames
            except soundfile.LibsndfileError as sE:
                print(sE)
                continue
            line = "{}\t{}".format(os.path.relpath(file_path, dir_path), frames)
            print(line, file=pretrain_f_lof)

    # pretrain files first
    pretrain_f = open(os.path.join(args.dest, "pretrain.tsv"), "w")
    print(dir_path, file=pretrain_f)  # header
    for fname in (
        files_without_labels
    ):  # Second Pass for files without labels. They go into pretrain
        file_path = os.path.realpath(fname)

        try:
            frames = soundfile.info(file_path).frames
        except soundfile.LibsndfileError as sE:
            print(sE)
            continue
        line = "{}\t{}".format(os.path.relpath(file_path, dir_path), frames)
        print(line, file=pretrain_f)

    if not args.valid_set_individuals:
        sss = MultilabelStratifiedShuffleSplit(
            n_splits=args.n_split, test_size=args.valid_percent, random_state=args.seed
        )
        it_ = (
            sss.split(labels_X, targets)
            if 0 < args.valid_percent
            else (np.arange(len(labels_X)), [])
        )
    else:
        it_ = split_train_valid_by_individuals(
            labels_X, args.valid_set_individuals, args.id_lookup_file_path
        )
    if len(labels_X) > 0 and len(targets) > 0:
        for idx, (train_index, test_index) in enumerate(it_):
            str_args = (idx + 1, args.n_split, len(train_index), len(test_index))
            print(
                "Pass {:01.0f} of {}, {:06.0f} train and {:06.0f} test samples".format(
                    *str_args
                )
            )
            if args.few_shot and 0 < args.valid_percent:
                print(
                    "We are adding five few-shot manifest files to current train split"
                )
                few_x_labels = list(np.array(labels_X)[train_index])
                few_y_targets = list(np.array(targets)[train_index])
                few_shot_generators = []
                for vp in [0.01, 0.1, 0.25, 0.5, 0.75]:
                    few_1 = MultilabelStratifiedShuffleSplit(
                        n_splits=1, test_size=1.0 - vp, random_state=args.seed
                    )
                    # just take the train split
                    few_train_indices = list(few_1.split(few_x_labels, few_y_targets))[
                        0
                    ][0]
                    # save the indices of the actual train split
                    few_shot_generators.append(train_index[few_train_indices])

            train_f = open(os.path.join(args.dest, "train_{}.tsv".format(idx)), "w")
            if args.few_shot and 0 < args.valid_percent:
                train_few_0 = open(
                    os.path.join(args.dest, "train_{}_few_0.tsv".format(idx)), "w"
                )
                train_few_1 = open(
                    os.path.join(args.dest, "train_{}_few_1.tsv".format(idx)), "w"
                )
                train_few_2 = open(
                    os.path.join(args.dest, "train_{}_few_2.tsv".format(idx)), "w"
                )
                train_few_3 = open(
                    os.path.join(args.dest, "train_{}_few_3.tsv".format(idx)), "w"
                )
                train_few_4 = open(
                    os.path.join(args.dest, "train_{}_few_4.tsv".format(idx)), "w"
                )
                print(dir_path, file=train_few_0)  # header
                print(dir_path, file=train_few_1)  # header
                print(dir_path, file=train_few_2)  # header
                print(dir_path, file=train_few_3)  # header
                print(dir_path, file=train_few_4)  # header

            valid_f = (
                open(os.path.join(args.dest, "valid_{}.tsv".format(idx)), "w")
                if args.valid_percent > 0
                else None
            )

            print(dir_path, file=train_f)  # header
            if valid_f is not None:
                print(dir_path, file=valid_f)  # header

            for (
                filename_index
            ) in train_index:  # Second Pass for train files with labels
                file_path = os.path.realpath(labels_X[filename_index])

                try:
                    frames = soundfile.info(file_path).frames
                except soundfile.LibsndfileError as sE:
                    print(sE)
                    continue
                line = "{}\t{}".format(os.path.relpath(file_path, dir_path), frames)
                print(line, file=train_f)
                if args.few_shot and 0 < args.valid_percent:
                    if filename_index in few_shot_generators[0]:
                        print(line, file=train_few_0)
                    if filename_index in few_shot_generators[1]:
                        print(line, file=train_few_1)
                    if filename_index in few_shot_generators[2]:
                        print(line, file=train_few_2)
                    if filename_index in few_shot_generators[3]:
                        print(line, file=train_few_3)
                    if filename_index in few_shot_generators[4]:
                        print(line, file=train_few_4)
                if pretrain_f is not None:
                    print(line, file=pretrain_f)

            if valid_f is not None:
                for (
                    filename_index
                ) in test_index:  # Second Pass for test files with labels
                    file_path = os.path.realpath(labels_X[filename_index])

                    try:
                        frames = soundfile.info(file_path).frames
                    except soundfile.LibsndfileError as sE:
                        print(sE)
                        continue
                    line = "{}\t{}".format(os.path.relpath(file_path, dir_path), frames)
                    print(line, file=valid_f)
                    if pretrain_f is not None:
                        print(line, file=pretrain_f)

            train_f.close()
            if valid_f is not None:
                valid_f.close()
            if (
                pretrain_f is not None
            ):  # Close pretrain file after first train/valid split, it has already everything
                pretrain_f.close()
                pretrain_f = None
            if args.few_shot and 0 < args.valid_percent:
                train_few_0.close()
                train_few_1.close()
                train_few_2.close()
                train_few_3.close()
                train_few_4.close()
    if pretrain_f is not None:
        pretrain_f.close()


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(args)
