import streamlit as st
import pandas as pd
from utils import clean_text

def render_batch_page(pipe, db):
    """Render the Batch Upload tab"""
    st.subheader('Bulk Analysis')
    
    # Download Samples
    st.markdown("### Download Sample Files")
    col1, col2 = st.columns(2)
    with col1:
        sample_csv = pd.DataFrame({'article_text': [
            'NASA has announced a groundbreaking new mission to explore the icy moons of Jupiter. The Europa Clipper spacecraft will conduct detailed reconnaissance of Jupiter\'s moon Europa and investigate whether the icy moon could harbor conditions suitable for life. The mission is set to launch next year and will take several years to reach its destination. Scientists are particularly interested in the subsurface ocean that is believed to exist beneath Europa\'s icy crust. This ocean could potentially contain more water than all of Earth\'s oceans combined.',
            'A recent medical research study published in a leading journal has highlighted the significant benefits of regular cardiovascular exercise on long-term heart health. The comprehensive study, which followed thousands of participants over a decade, found that individuals who engaged in at least 150 minutes of moderate-intensity aerobic activity per week had a markedly lower risk of developing heart disease. Furthermore, the research indicated that consistent exercise also contributes to better mental well-being and improved cognitive function in older adults.',
            'The global transition towards renewable energy sources is accelerating, with solar and wind power leading the charge. Several countries have recently reported record-breaking levels of energy generation from these sustainable sources, significantly reducing their reliance on fossil fuels. This shift is not only crucial for combating climate change but also for creating new economic opportunities and green jobs. Innovations in energy storage technology, such as advanced battery systems, are further enhancing the reliability and efficiency of renewable energy grids.',
            'Whistleblowers have revealed a shocking alien government conspiracy that has been kept hidden from the public for decades. According to leaked documents, extraterrestrial beings have been collaborating with high-ranking officials to control global events and suppress advanced technologies. These secret alliances are supposedly responsible for numerous unexplained phenomena and cover-ups. The documents suggest that the governments of major world powers have been exchanging resources for advanced alien weaponry and mind-control devices.',
            'A miraculous new cure for all types of cancer has been discovered, but Big Pharma is desperately trying to keep it hidden from the public to protect their massive profits. This revolutionary natural remedy, derived from a rare plant found deep within the Amazon rainforest, has reportedly cured hundreds of patients in secret trials. Despite overwhelming evidence of its effectiveness, regulatory agencies, influenced by pharmaceutical lobbyists, refuse to approve the treatment. Proponents urge the public to seek out this hidden cure before it is completely suppressed.'
        ]}).to_csv(index=False).encode('utf-8')
        st.download_button("Download Sample CSV", sample_csv, "sample.csv", "text/csv")
    with col2:
        sample_txt = """=== ARTICLE ===
NASA has announced a groundbreaking new mission to explore the icy moons of Jupiter. The Europa Clipper spacecraft will conduct detailed reconnaissance of Jupiter's moon Europa and investigate whether the icy moon could harbor conditions suitable for life. The mission is set to launch next year and will take several years to reach its destination. Scientists are particularly interested in the subsurface ocean that is believed to exist beneath Europa's icy crust. This ocean could potentially contain more water than all of Earth's oceans combined.

=== ARTICLE ===
A recent medical research study published in a leading journal has highlighted the significant benefits of regular cardiovascular exercise on long-term heart health. The comprehensive study, which followed thousands of participants over a decade, found that individuals who engaged in at least 150 minutes of moderate-intensity aerobic activity per week had a markedly lower risk of developing heart disease. Furthermore, the research indicated that consistent exercise also contributes to better mental well-being and improved cognitive function in older adults.

=== ARTICLE ===
The global transition towards renewable energy sources is accelerating, with solar and wind power leading the charge. Several countries have recently reported record-breaking levels of energy generation from these sustainable sources, significantly reducing their reliance on fossil fuels. This shift is not only crucial for combating climate change but also for creating new economic opportunities and green jobs. Innovations in energy storage technology, such as advanced battery systems, are further enhancing the reliability and efficiency of renewable energy grids.

=== ARTICLE ===
Whistleblowers have revealed a shocking alien government conspiracy that has been kept hidden from the public for decades. According to leaked documents, extraterrestrial beings have been collaborating with high-ranking officials to control global events and suppress advanced technologies. These secret alliances are supposedly responsible for numerous unexplained phenomena and cover-ups. The documents suggest that the governments of major world powers have been exchanging resources for advanced alien weaponry and mind-control devices.

=== ARTICLE ===
A miraculous new cure for all types of cancer has been discovered, but Big Pharma is desperately trying to keep it hidden from the public to protect their massive profits. This revolutionary natural remedy, derived from a rare plant found deep within the Amazon rainforest, has reportedly cured hundreds of patients in secret trials. Despite overwhelming evidence of its effectiveness, regulatory agencies, influenced by pharmaceutical lobbyists, refuse to approve the treatment. Proponents urge the public to seek out this hidden cure before it is completely suppressed.""".encode('utf-8')
        st.download_button("Download Sample TXT", sample_txt, "sample.txt", "text/plain")
        
    st.markdown("---")
    
    uploaded_file = st.file_uploader("Upload File (CSV or TXT)", type=['csv', 'txt'])
    
    if uploaded_file is not None:
        file_ext = uploaded_file.name.split('.')[-1].lower()
        articles = []
        df = None
        
        try:
            if file_ext == 'txt':
                content = uploaded_file.read().decode('utf-8').strip()
                if not content:
                    st.error("The uploaded TXT file is empty.")
                    return
                
                articles = [
                    article.strip()
                    for article in content.split("=== ARTICLE ===")
                    if article.strip()
                ]
                
                st.info(f"TXT File Uploaded | {len(articles)} Articles Detected")
                
            elif file_ext == 'csv':
                df = pd.read_csv(uploaded_file)
                if df.empty:
                    st.error("The uploaded CSV file is empty.")
                    return
                
                # Detect columns
                possible_cols = ['text', 'article', 'content', 'news', 'article_text']
                found_cols = [c for c in df.columns if c.lower() in possible_cols]
                
                text_col = None
                if len(found_cols) == 0:
                    st.error(f"No suitable text column found. Please ensure your CSV has one of: {', '.join(possible_cols)}")
                    return
                elif len(found_cols) == 1:
                    text_col = found_cols[0]
                else:
                    text_col = st.selectbox("Multiple text columns detected. Please select the one containing the articles:", found_cols)
                
                if text_col:
                    raw_len = len(df)
                    df = df.dropna(subset=[text_col])
                    articles = df[text_col].astype(str).str.strip().tolist()
                    articles = [a for a in articles if len(a) > 50]
                    skipped = raw_len - len(articles)
                    st.info(f"File type: CSV | {len(articles)} articles detected | {skipped} empty/short rows skipped.")
            else:
                st.error("Unsupported file format.")
                return
                
            if not articles:
                st.error("No valid articles found in the uploaded file. Check that the content is long enough (>50 characters).")
                return
                
            # Preview first 3 articles
            st.markdown("### Preview")
            for i, art in enumerate(articles[:3]):
                st.write(f"**Article {i+1}:** {art[:150]}...")
                
            if st.button('Start Batch Analysis'):
                results = []
                progress_bar = st.progress(0)
                
                for i, text in enumerate(articles):
                    cleaned = clean_text(text)
                    proba = pipe.predict_proba([cleaned])[0]
                    pred = 'FAKE' if proba[1] > 0.5 else 'REAL'
                    conf = max(proba) * 100
                    
                    results.append({
                        'text_preview': text[:100] + '...',
                        'prediction': pred,
                        'confidence': f"{conf:.1f}%",
                        'real_prob': proba[0],
                        'fake_prob': proba[1]
                    })
                    
                    # Save to DB if available
                    if db:
                        try:
                            db.add_analysis(
                                article_text=text,
                                prediction=pred,
                                confidence=conf,
                                real_prob=proba[0],
                                fake_prob=proba[1],
                                red_flag_score=0.0
                            )
                        except:
                            pass
                    
                    progress_bar.progress((i + 1) / len(articles))
                
                st.success("Analysis Complete!")
                results_df = pd.DataFrame(results)
                st.dataframe(results_df)
                
                # Download button
                csv = results_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    "Download Results CSV",
                    csv,
                    "analysis_results.csv",
                    "text/csv",
                    key='download-csv'
                )
                    
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
