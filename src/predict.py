import numpy as np

class EmotionPredictor:
    def __init__(self, classes):
        self.classes = classes

        # FIX: removed overly generic words that caused false positives:
        # - 'tried' (Frustrated) -- appears in tons of neutral sentences
        # - 'good', 'great' (Confident) -- extremely common filler words
        # ADDED: 'confusing' to Confused (was missing entirely before)
        self.emotion_keywords = {
            'Frustrated': ['frustrated', 'frustrating', 'annoying', 'angry', 'hate',
                           'difficult', 'stuck', 'wrong answer', 'keep getting',
                           'unnecessarily complicated'],
            'Curious': ['why', 'how', 'what', 'curious', 'wonder', 'interested',
                        'learn', 'know more', 'want to know', 'explore',
                        'could we', 'what happens', 'intuition', 'behind'],
            'Confident': ['easy', 'amazing', 'excellent', 'awesome',
                          'perfect', 'solved', 'got it', 'clear now', 'finally',
                          'move ahead', 'understand clearly', 'makes sense now'],
            'Bored': ['boring', 'bored', 'tired', 'repetitive', 'dull', 'not engaging',
                      "didnt feel engaging", "didn't feel engaging", 'not interesting',
                      'too basic', 'losing'],
            'Confused': ['confused', 'confusing', 'lost', 'unclear', 'dont understand',
                         "don't understand", "doesn't make sense", 'not fully confident',
                         'missing', 'incomplete', 'unsure']
        }

    def boost_with_keywords(self, text, probs):
        text_lower = text.lower()
        probs = np.array(probs, dtype=float)

        emotion_scores = {}
        for emotion, keywords in self.emotion_keywords.items():
            score = 0
            for keyword in keywords:
                if keyword in text_lower:
                    if keyword in ['frustrated', 'frustrating', 'curious', 'confident', 'bored', 'boring', 'confused']:
                        score += 10
                    else:
                        score += 2
            emotion_scores[emotion] = score

        max_score = max(emotion_scores.values())
        if max_score > 0:
            winning_emotions = [e for e, s in emotion_scores.items() if s == max_score]

            # Gentler boost so mixed emotions survive (see earlier fix)
            boost_factor = 1.8
            suppress_factor = 0.85

            for emotion in winning_emotions:
                idx = self.classes.index(emotion)
                probs[idx] *= boost_factor

            for i, emotion in enumerate(self.classes):
                if emotion not in winning_emotions:
                    probs[i] *= suppress_factor

        probs = probs / np.sum(probs)
        return probs


if __name__ == "__main__":
    classes = ['Bored', 'Confident', 'Confused', 'Curious', 'Frustrated']
    booster = EmotionPredictor(classes)
    fake_probs = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    for text in ["This is a good question", "This topic is confusing", "I understand clearly now"]:
        result = booster.boost_with_keywords(text, fake_probs.copy())
        print(text, "->", dict(zip(classes, result)))