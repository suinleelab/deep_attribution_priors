import torch

class StaticFeatureAttributionPrior:
    def __init__(self, explainer, prior_feature, ignored_features, background_dataset):
        super(StaticFeatureAttributionPrior, self).__init__()
        self.explainer = explainer
        self.prior_feature = prior_feature
        self.ignored_features = ignored_features
        self.background_dataset = background_dataset

    def penalty(self, model, features):
        prior_feature = torch.FloatTensor(self.prior_feature).cuda()
        explainer_for_batch = self.explainer(self.background_dataset, features.shape[0], k=2)
        shap_values = explainer_for_batch.shap_values(model, features)
        eg_no_drugs = shap_values[:, len(self.ignored_features):]
        attribution_diff = eg_no_drugs - prior_feature

        #FIXME: Hardcoded 20 here for SCORE feature
        prior_penalty = (attribution_diff ** 2).mean() / 20

        return prior_penalty
