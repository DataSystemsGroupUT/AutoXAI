from sklearn.linear_model import LogisticRegression
from autoxai.data_loader import load_sample_dataset
from autoxai.recommender import AutoXAIRecommender

def main():
    X, y = load_sample_dataset()
    model = LogisticRegression(max_iter=1000).fit(X, y)

    recommender = AutoXAIRecommender()
    recommender.fit(model, X, y)
    
    print("Recommended explanation methods (top 2):")
    print(recommender.recommend(top_k=2))

if __name__ == "__main__":
    main()
