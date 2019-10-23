class StaticFeatureAttributionPrior:
    def __init__(self, explainer, prior_feature, ignored_features):
        super(StaticFeatureAttributionPrior, self).__init__()
        self.explainer = explainer
        self.prior_feature = prior_feature
        self.ignored_features = ignored_features

    def penalty(self, model, features, alpha):
        # Short circuit and don't calculate feature attributions if we don't need them
        if alpha == 0:
            return 0

        shap_values = self.explainer.shap_values(model, features)
        eg_no_drugs = shap_values[:, len(self.ignored_features):]
        attribution_diff = eg_no_drugs - self.prior_feature
        prior_penalty = alpha*(attribution_diff ** 2).mean()

        return alpha * prior_penalty
