from rest_framework.decorators import api_view
from rest_framework.response import Response
from textblob import TextBlob

@api_view(["POST"])
def predict_review(request):
    review = request.data.get("review", "")

    if not review.strip():
        return Response({
            "prediction": "No review provided",
            "confidence": 0
        })

    blob = TextBlob(review)
    polarity = blob.sentiment.polarity
    subjectivity = blob.sentiment.subjectivity

    exaggerated_words = [
        "amazing", "perfect", "best", "excellent", "life changing",
        "must buy", "buy now", "100% recommended", "five stars",
        "awesome", "fantastic", "unbelievable"
    ]

    review_lower = review.lower()
    exaggeration_count = sum(1 for word in exaggerated_words if word in review_lower)

    if subjectivity > 0.65 and polarity > 0.55 and exaggeration_count >= 1:
        prediction = "Fake Review"
        confidence = min(95, 60 + exaggeration_count * 10 + subjectivity * 20)
    else:
        prediction = "Real Review"
        confidence = min(90, 55 + (1 - subjectivity) * 25)

    return Response({
        "prediction": prediction,
        "confidence": round(confidence, 2)
    })
