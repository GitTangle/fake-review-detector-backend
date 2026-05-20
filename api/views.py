import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status

MODEL_NAME = "distilbert-base-uncased"

tokenizer = DistilBertTokenizer.from_pretrained(MODEL_NAME)
model = DistilBertForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=2
)
model.eval()

LABELS = {
    0: "Real Review",
    1: "Fake / AI Generated Review"
}

@api_view(["POST"])
def predict_review(request):
    try:
        text = request.data.get("review", "").strip()

        if not text:
            return Response(
                {"error": "No review text provided"},
                status=status.HTTP_400_BAD_REQUEST
            )

        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=256
        )

        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.softmax(outputs.logits, dim=1)
            pred = torch.argmax(probs, dim=1).item()
            confidence = round(probs[0][pred].item() * 100, 2)

        return Response({
            "review": text,
            "prediction": LABELS[pred],
            "confidence": f"{confidence}%"
        })

    except Exception as e:
        return Response(
            {"error": str(e)},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )