def print_result(TP, FP, FN, TN):
    beta = 1
    accuracy = (TP + TN) / (TP + TN + FP + FN + 1e-8)
    positive_precision = TP / (TP + FP + 1e-8)
    positive_recall = TP / (TP + FN + 1e-8)
    positive_f = (1 + beta ** 2) * positive_precision * positive_recall / (
            (beta ** 2) * positive_precision + positive_recall + 1e-8)
    negative_precision = TN / (TN + FN + 1e-8)
    negative_recall = TN / (TN + FP + 1e-8)
    negative_f = (1 + beta ** 2) * negative_precision * negative_recall / (
            (beta ** 2) * negative_precision + negative_recall + 1e-8)
    print(f'TP:{TP}, FP:{FP}')
    print(f'FN:{FN}, TN:{TN}')
    print("accuracy:-----------{:.6f}".format(accuracy))
    print("positive precision:-{:.6f}".format(positive_precision))
    print("positive recall:----{:.6f}".format(positive_recall))
    print("positive f{:.1f}:------{:.6f}".format(beta, positive_f))
    print("negative precision:-{:.6f}".format(negative_precision))
    print("negative recall:----{:.6f}".format(negative_recall))
    print("negative f{:.1f}:------{:.6f}".format(beta, negative_f))
    return positive_f