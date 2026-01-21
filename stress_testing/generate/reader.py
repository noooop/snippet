


import numpy as np


def main(filename: str, log_interval:int=100):
    metrics = np.load(filename)
    b, n = metrics.shape
    sorted_metrics = np.sort(metrics, axis=0)

    b10 = int(b * 0.1)
    b20 = int(b * 0.2)
    b25 = int(b * 0.25)
    b50 = int(b * 0.5)
    b75 = int(b * 0.75)
    b80 = int(b * 0.8)
    b90 = int(b * 0.9)

    for i, (a, b, c, d, e, f, g) in enumerate(zip(sorted_metrics[b10],
                                                  sorted_metrics[b20],
                                                  sorted_metrics[b25],
                                                  sorted_metrics[b50],
                                                  sorted_metrics[b75],
                                                  sorted_metrics[b80],
                                                  sorted_metrics[b90])
                                              ):
        if i%log_interval==0:
            print(i, d*1000)

if __name__ == "__main__":
    import sys
    filename = sys.argv[1]
    main(filename)