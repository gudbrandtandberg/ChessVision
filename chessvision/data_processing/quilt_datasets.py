import quilt
import cv_globals
import os

dataset_name = "gudbrandtandberg/chesspieces"

hashes = [
    "25dc6b55d223e549d757fa2414dc466fd7c03ffe9578b80bac07f99138a9dffc",
    "35f4c4878d83072de61c6d7bb656223255dd26152962e2f01b8404dd93be1b3c",
    "3f4cc6de6fce6a429eb191aa3911309ab9d2baf789735b50fc858414cabb3404",
    "e15236eddf272c1f682dca4607f09e826a69bbfa58afc943507b80b5e5c96187",
    "0d4f4c8bd90fcd9d654b569cafb2b55fbf2a2555cca84292feb1f070045ced10",
    "dd120ebfa18100bfc523f435fdae860c8549834640d0b237a08b70607662b1c8",
    "096c5feb9b2ad469b524d47e0a7971de5835b5ca844d5a1732423168a9d90857",
    "0c9055b8b010704b02a6a8dea45eb50f4904c6d97d8ac7258fc2cfff305620b5",
    "f54df2a1b551ee55c662f3aa1ef38f5eebac136f0afb721a94e0dbc98ac104b7",
    "7f0bc7a0d3e409a05e37df3778e40fb6c33fd16d7e4b10292dcdebdf08d63c58",
    "a5b8ab199bb1eba24ab4b0a702d7f1554c38e5afafc8629e10d339ef11318bf4",
    "3a9ec28cf5968aae4bd8c03ad2e6d14932864874e9f0e2f25f33095af459b082",
    "bae951a8154c86c5bd83fc97ed04d9d3441a88505250019dc9af3004b5d33c32",
    "4a35cd093dc3183d7098da6a009dedde4e3ef837f21d40bf0c5e2a44f90d1295",
    "c8469b9d04e92fea3ff03a344cb512988783cdfa17f6336a35c31e07a4668b02",
]

realized_hashes = [
    (0, "25dc6b55d223e549d757fa2414dc466fd7c03ffe9578b80bac07f99138a9dffc"),
    (1, "35f4c4878d83072de61c6d7bb656223255dd26152962e2f01b8404dd93be1b3c"),
    (9, "7f0bc7a0d3e409a05e37df3778e40fb6c33fd16d7e4b10292dcdebdf08d63c58"),
    (10, "a5b8ab199bb1eba24ab4b0a702d7f1554c38e5afafc8629e10d339ef11318bf4"),
    (11, "3a9ec28cf5968aae4bd8c03ad2e6d14932864874e9f0e2f25f33095af459b082"),
    (12, "bae951a8154c86c5bd83fc97ed04d9d3441a88505250019dc9af3004b5d33c32"),
    (13, "4a35cd093dc3183d7098da6a009dedde4e3ef837f21d40bf0c5e2a44f90d1295"),
    (14, "c8469b9d04e92fea3ff03a344cb512988783cdfa17f6336a35c31e07a4668b02"),
]

def export_revisions():
    for index, hash in enumerate(hashes):
        try:
            quilt.export(dataset_name, output_path=f"{cv_globals.CVROOT}/data/quilt_revisions/{index}_{hash}", hash=hash)
        except quilt.tools.command.CommandException as e:
            print(f"Already exported {hash}, skipping.. ({e})")

def get_filenames_from_revision(path):
    all_files = []
    for root, dirs, files in os.walk(path):
        stripped_root = root.replace(path, "").lstrip("\\")
        for f in files:
            all_files.append(os.path.normpath(os.path.join(stripped_root, f)))
    return all_files
            

def get_revision_diff(hash1, hash2):
    """Find addition, moves, removals.

    Read all files in 1 and 2

    iterate over 2:
        if in 1 with same path: pass
        if in 1 with different path: moved
        if not in 1: added

    iterate over 1:
        if in pass, moved, added: skip
        else: removed
    
    Returns:
        added, moved (old, new), removed
    """

    files_1 = get_filenames_from_revision(f"{cv_globals.CVROOT}/data/quilt_revisions/{hash1}")
    files_2 = get_filenames_from_revision(f"{cv_globals.CVROOT}/data/quilt_revisions/{hash2}")

    # print(f"Comparing revisions with {len(files_2)} and {len(files_1)} entries")

    files_1_names = list(map(lambda x: x.split(os.sep)[-1], files_1))

    ADDED = []
    REMOVED = []
    MOVED = []
    NO_CHANGE = []

    for f2 in files_2:
        fname = f2.split(os.sep)[-1]
        if f2 in files_1:
            NO_CHANGE.append(f2)
        else:
            if fname in files_1_names:
                res = list(filter(lambda x: fname in x, files_1))
                assert len(res) == 1
                MOVED.append((res[0], f2))
            else:
                ADDED.append(f2)
    
    moved_destinations = [m[1] for m in MOVED]
    moved_names = list(map(lambda x: x.split(os.sep)[-1], moved_destinations))
    for f1 in files_1:
        if f1 not in moved_destinations and f1 not in ADDED and f1 not in NO_CHANGE:
            f1name = f1.split(os.sep)[-1]
            if f1name not in moved_names:
                REMOVED.append(f1)
        
    
    assert len(files_2) == len(files_1) + len(ADDED) - len(REMOVED)
    return ADDED, REMOVED, MOVED

def get_revision_diffs():
    
    for i in range(len(realized_hashes) - 1):
        this = realized_hashes[i]
        next = realized_hashes[i+1]

        print(f"======== Comparing revision {this[0]} and {next[0]}")
        added, removed, moved = get_revision_diff(f"{this[0]}_{this[1]}", f"{next[0]}_{next[1]}")
        print(f"\tAdded {len(added)} items, removed {len(removed)} items, moved {len(moved)} items")


if __name__ == "__main__":
    # export_revisions()
    get_revision_diffs()


