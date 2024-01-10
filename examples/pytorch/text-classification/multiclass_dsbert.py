# from transformers import DistilBertForMultiLabelClassification
from transformers.models.distilbert.configuration_distilbert import DistilBertForMultiLabelConfig
from transformers.models.distilbert.modeling_distilbert_multilabel import DistilBertForMultiLabelClassification
from transformers import DistilBertModel

config = DistilBertForMultiLabelConfig()
# set number of classes
config.num_classes_per_level = [15, 140]
print(config)

model = DistilBertForMultiLabelClassification(config=config)
# load pretrained distilbert model
model.bert = DistilBertModel.from_pretrained("distilbert-base-uncased")

