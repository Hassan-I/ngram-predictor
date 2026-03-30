import streamlit as st

class PredictorUI:
    def __init__(self, predictor, top_k):
        self.predictor = predictor
        self.top_k = top_k
        

    def apply_custom_style(self):
        """Adds custom CSS to make it look modern."""
        st.markdown("""
            <style>
            .main {
                background-color: #f5f7f9;
            }
            .stTextInput intput {
                border-radius: 20px;
            }
            .prediction-badge {
                display: inline-block;
                padding: 8px 16px;
                margin: 5px;
                background-color: #e1f5fe;
                border: 1px solid #01579b;
                border-radius: 15px;
                color: #01579b;
                font-weight: bold;
            }
            </style>
        """, unsafe_allow_html=True)

    def render_header(self):
        # Using a container for better spacing
        with st.container():
            st.title("🔮 N-Gram Language Model")
            st.info("The model suggests the most likely next word based on your current input.")
            st.divider()

    def render_sidebar(self):
        """Added a sidebar to show model metadata."""
        with st.sidebar:
            st.header("🛠 Settings")
            st.write("Adjusting these values updates the UI instantly.")
            # We use session state to allow the UI to override default top_k
            self.top_k = st.slider("Number of predictions (K)", 1, 10, self.top_k)
            
            st.divider()
            st.caption("Powered by N-Gram Smoothing Model")

    def render_input(self):
        # Group input in a clean box
        st.subheader("Input Sequence")
        text = st.text_input(
            label="Type something below:",
            placeholder="e.g. 'The quick brown fox'",
            label_visibility="collapsed"
        )
        return text

    def render_predictions(self, predictions):
        st.divider()
        if predictions:
            st.subheader("🎯 Next Word Suggestions")
            
            # Display predictions as horizontal "badges" using columns
            cols = st.columns(len(predictions))
            for i, word in enumerate(predictions):
                with cols[i]:
                    # Using success style for a 'chip' look
                    st.button(word, key=f"btn_{i}", use_container_width=True)
        else:
            st.warning("⚠️ No predictions available. Try typing more common words.")

    def run(self):
        self.apply_custom_style()
        self.render_sidebar()
        self.render_header()
        
        text = self.render_input()
        
        if text.strip():
            with st.spinner("Analyzing context..."):
                predictions = self.predictor.predict_next(text, self.top_k)
                self.render_predictions(predictions)
        else:
            st.write("Please enter some text to see predictions.")