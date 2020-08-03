class Measure:
    def __init__(self, acc_score, prec_score, recall_measure, f1_measure):
        self.acc_score = '{0:.8g}'.format(100*acc_score)
        self.prec_score = '{0:.8g}'.format(100*prec_score)
        self.recall_measure = '{0:.8g}'.format(100*recall_measure)
        self.f1_measure = '{0:.8g}'.format(100*f1_measure)

    def getObj(self):
        return {
            'accuracy': self.acc_score,
            'precision': self.prec_score,
            'recall': self.recall_measure,
            'f1_score': self.f1_measure
        }
