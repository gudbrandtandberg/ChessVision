# Chess Vision - a computer vision chess project


## Board extraction

## Board rotation

### Square classification

I will attempt classifying squares by retraining a pretrained CNN on a hand-labelled dataset.

There are 12 different pieces on a chess board. They are:

  white: R, N, B, Q, K, P 
  black: r, n, b, q, k, p 

In addition, any square can be free (f).
Any piece can be posted on a white (w) or dark (d) square.

If we do not take into account square color, this turns out to be 13 classes.

If we take into account square color, this turns out to be 26 classes.

I suspect there might be a accuracy trade-off here, but will attempt 13-class classification first.

I will write a script that lets me hand-label every square image in my dataset into one of the 13 categories. 

Forever:
    show image
    wait for user-input... 
    record label
