import papermill as pm
import cv_globals
import argparse
import subprocess
import os

if __name__ == "__main__":
    # Example usage:
    # python generate_report.py --extractor_weights=/Users/gudbrand/Programming/Chess/ChessVision/weights/new_extractor.hdf5 --report_name=new_extractor_report
    # python generate_report.py --classifier_weights=/Users/gudbrand/Programming/Chess/ChessVision/weights/new_classifier.hdf5 --report_name=new_classifier_report

    parser = argparse.ArgumentParser(
        description='Output a html report of chessvision performance')
    parser.add_argument("--extractor_weights", type=str, default=cv_globals.board_weights)
    parser.add_argument("--classifier_weights", type=str, default=cv_globals.square_weights)
    parser.add_argument("--threshold", type=int, default=80)
    parser.add_argument("--report_name", type=str, default="new_report")
    args = parser.parse_args()

    infile  = cv_globals.CVROOT + "/chessvision/notebooks/Performance Report.ipynb"
    outfile = cv_globals.CVROOT + "/chessvision/notebooks/{}.ipynb".format(args.report_name)
    
    pm.execute_notebook(
        infile,
        outfile,
        parameters=dict(extractor_weights=args.extractor_weights,
                        classifier_weights=args.classifier_weights, 
                        threshold=args.threshold)
    )

    subprocess.call(["jupyter-nbconvert", outfile])
    os.remove(outfile)