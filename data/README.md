# Dataset

The dataset was collected from a combination of photos of chessbooks taken from the local library together with photos of chessboards on a computer screen. All the images are cropped to a square aspect ratio. The original raw dataset is not contained in this repository because it is too large. 

Instead, all the processed training data acquired from this app ends up in this repository. 

Data for training the board extractor is contained in [board_extraction](./board_extraction). This dataset consists of hundreds of 256x256 square images together with chessboard masks. The masks are acquired manually using [dataturks](https://dataturks.com) (send me a message if you'd like to help classifying chessboards!). 

For the piece classifier, data belonging to the 13 classes B, K, Q, R, N, b, k, q, r, n, f, (extracted from boards using either labelled masks or board extraction), is stored in the [squares](./squares) directory.

Scripts for processing data, including pipeline and transformation, lie in [data_processing](../chessvision/data_processing).
