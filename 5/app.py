from flask import Flask, render_template, request, jsonify
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

app = Flask(__name__)

class SimpleChatbot:
    def __init__(self):
        self.responses = {
            "hello": "Hello! Welcome to Amazon. How can I assist you today?",
            "bye": "Thank you for shopping with Amazon. Have a great day!",
            "product_info": "Our top-selling product is the Amazon Echo. It's a smart speaker with Alexa voice control.",
            "thanks": "You're welcome! If you have any more questions, feel free to ask.",
            "order_status": "To check your order status, please provide your order number or login to your Amazon account.",
            "delivery_info": "Our standard delivery time is 2-3 business days. You can also choose expedited shipping for faster delivery.",
            "return_policy": "We have a hassle-free return policy. You can return most items within 30 days for a full refund.",
            "payment_options": "We accept various payment options including credit/debit cards, Amazon Pay, and PayPal.",
            "prime_membership": "Consider joining Amazon Prime for free shipping, exclusive deals, and access to Prime Video and Music.",
            "customer_service": "Our customer service team is available 24/7 to assist you with any questions or concerns.",
            "gift_cards": "Looking for a gift? Consider an Amazon gift card, the perfect present for any occasion.",
            "wishlist": "Create a wishlist to keep track of items you want to buy later or share with friends and family.",
            "deals_and_promotions": "Check out our deals and promotions page for discounts on a wide range of products.",
            "recommendations": "Based on your browsing and purchase history, here are some personalized recommendations for you.",
            "tech_support": "For technical support with your Amazon device or service, please visit our support center or contact us.",
            "security_and_privacy": "We take security and privacy seriously. Learn more about how we protect your information.",
            "shopping_cart": "Your shopping cart contains items ready for purchase. Proceed to checkout to complete your order.",
            "order_tracking": "Track your package in real-time with our order tracking system. Just enter your order number.",
            "subscription_services": "Subscribe and save on everyday essentials with Amazon's subscription services.",
            "international_shipping": "We offer international shipping to select countries. Check if your country is eligible for delivery.",
            "customer_reviews": "Read customer reviews to make informed buying decisions and share your own experience.",
            "feedback_survey": "Help us improve by taking our feedback survey after your shopping experience.",
            "lost_package": "If your package is lost or delayed, please contact our customer service team for assistance.",
            "business_accounts": "Explore Amazon Business for special pricing, multi-user accounts, and business-specific features.",
            "student_discounts": "Students can enjoy exclusive discounts and benefits with Amazon Student.",
            "product_comparison": "Compare products side by side to find the best option for your needs.",
            "gift_wrapping": "Add gift wrapping to your order for a special touch when sending gifts to loved ones.",
            "community_forum": "Join our community forum to connect with other shoppers, share tips, and ask questions.",
            "product_customization": "Some products offer customization options. Explore personalization features on product pages.",
            "voice_shopping": "Shop hands-free with Alexa voice shopping. Just ask Alexa to add items to your cart.",
            "virtual_assistant": "Get instant help and answers with our virtual assistant feature. Available 24/7.",
            "subscription_management": "Manage your subscriptions easily from your Amazon account. Add, remove, or update subscriptions as needed.",
            "amazon_sustainability": "Learn about Amazon's commitment to sustainability and our efforts to reduce our environmental impact.",
            "prime_day": "Don't miss out on Prime Day, our annual event with exclusive deals and discounts for Prime members.",
            "book_recommendations": "Discover new books and bestsellers with personalized book recommendations.",
            "amazon_fresh": "Shop for groceries and everyday essentials with Amazon Fresh. Fast delivery and wide selection available.",
            "live_support": "Chat live with a customer service representative for immediate assistance.",
            "product_availability": "Check product availability and delivery options in your area before placing an order.",
            "amazon_locker": "Use Amazon Locker for convenient delivery and pickup options at a location near you.",
            "automotive_parts": "Find the right automotive parts and accessories for your vehicle with our wide selection.",
            "charitable_donations": "Support charitable causes with AmazonSmile. A portion of your purchase goes to the charity of your choice.",
            "amazon_music": "Enjoy unlimited access to millions of songs with Amazon Music. Listen online or offline with a subscription.",
            "amazon_photos": "Store and share your photos securely with Amazon Photos. Free unlimited storage for Prime members.",
            "amazon_video": "Stream thousands of movies and TV shows with Amazon Prime Video. Exclusive content and original series available.",
            "smart_home_devices": "Transform your home with smart home devices from Amazon. Control lights, thermostats, and more with ease.",
            "amazon_games": "Discover a wide selection of games and gaming accessories with Amazon Games. From consoles to accessories, we have it all.",
            "tech_gadgets": "Stay up to date with the latest tech gadgets and accessories. Shop now for the newest releases.",
            "amazon_books": "Explore a vast selection of books in every genre. From bestsellers to classics, find your next read here.",
            "amazon_fashion": "Discover the latest fashion trends and styles with Amazon Fashion. Shop clothing, shoes, and accessories for every occasion.",
            "amazon_devices": "Shop Amazon devices for innovative technology and entertainment. From Kindle to Fire TV, find what you need.",
            "amazon_business": "Find solutions for your business needs with Amazon Business. Get special pricing, bulk discounts",
            "default": "I'm sorry, I didn't understand. Can you please rephrase?"
        }

        self.intent_classifier = MultinomialNB()
        self.vectorizer = CountVectorizer()

        # Training data
        self.train_data = list(self.responses.keys())
        self.train_labels = list(self.responses.keys())  # Using keys as labels

        # Preprocess and train
        self.X_train = self.vectorizer.fit_transform(self.train_data)
        self.intent_classifier.fit(self.X_train, self.train_labels)

    def classify_intent(self, user_input):
        user_input_vectorized = self.vectorizer.transform([user_input])
        intent = self.intent_classifier.predict(user_input_vectorized)
        return intent[0]

    def get_response(self, user_input):
        intent = self.classify_intent(user_input.lower())
        return self.responses.get(intent, self.responses["default"])

chatbot = SimpleChatbot()

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    user_input = request.json.get("user_input", "")
    response = chatbot.get_response(user_input)
    return jsonify({"response": response})

if __name__ == "__main__":
    app.run(debug=True)
