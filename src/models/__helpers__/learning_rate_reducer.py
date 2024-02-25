def learning_rate_reducer(loss, optimizer, lr_reduced):
    thresholds = [
        0.00027,
        0.00025,
        0.00020,
        0.00015,
        0.00013,
        0.00011,
        0.000090,
        0.000075,
        0.000050,
        0.000035,
        0.000025,
    ]

    for i, threshold in enumerate(thresholds):
        if loss.item() < threshold and not lr_reduced[i]:
            print("@@@@@@@@@@")
            print("@@@@@@@@@@")
            print("@@@@@@ Cutting learning_rate in half @@@@@@")
            print("@@@@@@@@@@")
            print("@@@@@@@@@@")
            for g in optimizer.param_groups:
                g["lr"] = g["lr"] / 2
            lr_reduced[i] = True

    return lr_reduced, optimizer
