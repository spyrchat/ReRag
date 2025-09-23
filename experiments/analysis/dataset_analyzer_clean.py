"""
SOSUM Dataset Analysis Tool for Master's Thesis
Creates publication-quality plots showcasing dataset characteristics
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import ast
import json
from collections import Counter, defaultdict
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

# Set up matplotlib for publication-quality plots
# Updated matplotlib settings to use Times New Roman as main font
plt.rcParams.update({
    'font.size': 11,
    'font.family': 'serif',  # Use serif as primary for Times New Roman
    # Times New Roman as primary font with fallbacks
    'font.serif': ['Times New Roman', 'DejaVu Serif', 'Liberation Serif'],
    # Your exact sans-serif
    'font.sans-serif': ['DejaVu Sans', 'Arial', 'Helvetica'],
    # Your exact monospace
    'font.monospace': ['DejaVu Sans Mono', 'Courier New'],
    'figure.dpi': 400,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.labelsize': 11,
    'axes.titlesize': 13,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 10,
    'figure.titlesize': 14,
    'axes.linewidth': 0.8,
    'xtick.major.width': 0.8,
    'ytick.major.width': 0.8,
    'grid.linewidth': 0.5,
    'axes.edgecolor': '#333333',
    'text.color': '#333333',
    # Better spacing to prevent cramping
    'axes.labelpad': 10,
    'axes.titlepad': 20,
    'xtick.major.pad': 6,
    'ytick.major.pad': 6,
    'legend.borderpad': 0.6,
    'legend.handletextpad': 0.8,
    'legend.columnspacing': 2.0
})

# Professional color palette matching academic papers
colors_thesis = {
    'primary': '#86579A',       # Academic purple
    'secondary': "#2E5984",     # Academic blue
    'accent': "#B34747",        # Academic red
    'neutral': '#E67E22',       # Academic orange
    'light': '#ECF0F1',         # Light gray
    'dark': '#2C3E50',          # Dark blue-gray
    'grid': '#BDC3C7',          # Subtle grid
    'text': '#2C3E50'           # Dark text for readability
}


class SOSUMDatasetAnalyzer:
    """Publication-quality analysis of SOSUM dataset for master's thesis."""

    def __init__(self, dataset_path: str):
        self.dataset_path = Path(dataset_path)
        self.question_file = self._find_csv_file("question.csv")
        self.answer_file = self._find_csv_file("answer.csv")

        # Create output directories
        self.plots_dir = Path("experiments/analysis/plots")
        self.plots_dir.mkdir(parents=True, exist_ok=True)

        # Load data with error handling
        print("🔍 Analyzing dataset structure...")
        self.questions_df = self._load_questions()
        self.answers_df = self._load_answers()

        print(
            f"📊 Loaded {len(self.questions_df)} questions and {len(self.answers_df)} answers")

    def _find_csv_file(self, filename: str) -> Path:
        """Find CSV file in dataset path or data subfolder."""
        direct_path = self.dataset_path / filename
        if direct_path.exists():
            return direct_path

        data_path = self.dataset_path / "data" / filename
        if data_path.exists():
            return data_path

        raise FileNotFoundError(
            f"Could not find {filename} in {self.dataset_path}")

    def _load_questions(self) -> pd.DataFrame:
        """Load and preprocess questions with SIMPLE answer counting."""
        df = pd.read_csv(self.question_file)

        print(f"📋 Question file columns: {list(df.columns)}")

        # Parse tags
        df['tags_parsed'] = df['tags'].apply(self._parse_tags)
        df['tag_count'] = df['tags_parsed'].apply(len)

        # SIMPLE: Just count items in answer_posts list
        df['answer_posts_parsed'] = df['answer_posts'].apply(
            self._parse_answer_posts)
        df['answer_count'] = df['answer_posts_parsed'].apply(
            len)  # Just count the list items!

        # Calculate text statistics
        df['title_length'] = df['question_title'].str.len()
        df['body_length'] = df['question_body'].apply(
            self._calculate_body_length)
        df['total_text_length'] = df['title_length'] + df['body_length']

        # Enhanced categorization using REAL tags
        df['primary_tag_category'] = df['tags_parsed'].apply(
            self._categorize_primary_tag)
        df['difficulty_level'] = df.apply(
            self._calculate_difficulty_level, axis=1)
        df['question_type_label'] = df['question_type'].map({
            1: 'Conceptual',
            2: 'How-to',
            3: 'Debug/Corrective'
        })

        return df

    def _parse_answer_posts(self, posts_str: str) -> List[str]:
        """Parse answer post IDs - SIMPLE VERSION."""
        if pd.isna(posts_str) or not str(posts_str).strip():
            return []

        try:
            posts_str = str(posts_str).strip()

            # Handle different formats
            if posts_str.startswith('[') and posts_str.endswith(']'):
                # It's a list format like ['id1', 'id2', 'id3']
                posts = ast.literal_eval(posts_str)
                if isinstance(posts, list):
                    return [str(post).strip() for post in posts if str(post).strip()]
            else:
                # It's comma-separated like 'id1,id2,id3'
                return [post.strip() for post in posts_str.split(',') if post.strip()]

        except Exception as e:
            print(f"⚠️  Error parsing '{posts_str}': {e}")
            return []

        return []

    def _load_answers(self) -> pd.DataFrame:
        """Load and preprocess answers with robust column detection."""
        df = pd.read_csv(self.answer_file)

        print(f"📋 Answer file columns: {list(df.columns)}")

        # Detect actual column names
        body_column = None
        summary_column = None

        # Common column name patterns
        for col in df.columns:
            col_lower = col.lower()
            if 'body' in col_lower:
                body_column = col
            elif 'summary' in col_lower:
                summary_column = col

        # If no body column found, use first non-ID column
        if body_column is None:
            non_id_cols = [
                col for col in df.columns if 'id' not in col.lower()]
            body_column = non_id_cols[0] if non_id_cols else df.columns[0]

        print(f"📊 Using '{body_column}' as body column")
        if summary_column:
            print(f"📊 Found summary column: '{summary_column}'")
        else:
            print("📊 No summary column found - will skip summary analysis")

        # Calculate text statistics
        df['body_length'] = df[body_column].apply(self._calculate_body_length)

        # Handle summary column if it exists
        if summary_column and summary_column in df.columns:
            df['has_summary'] = df[summary_column].notna() & (
                df[summary_column].str.strip() != '')
            df['summary_length'] = df[summary_column].apply(
                lambda x: self._calculate_body_length(x) if pd.notna(x) else 0)
        else:
            # Create dummy summary columns
            df['has_summary'] = False
            df['summary_length'] = 0
            print("⚠️  No summary data available - using dummy values")

        return df

    def _parse_tags(self, tags_str: str) -> List[str]:
        """Parse tags from string representation."""
        if pd.isna(tags_str) or not tags_str.strip():
            return []

        try:
            if tags_str.startswith('[') and tags_str.endswith(']'):
                tags = ast.literal_eval(tags_str)
                return [tag.strip().lower() for tag in tags if tag.strip()]
            else:
                return [tag.strip().lower() for tag in tags_str.split(',') if tag.strip()]
        except Exception as e:
            return []

    def _calculate_body_length(self, body: str) -> int:
        """Calculate length of body text (handling list format)."""
        if pd.isna(body):
            return 0

        try:
            if isinstance(body, str) and body.startswith('[') and body.endswith(']'):
                body_list = ast.literal_eval(body)
                if isinstance(body_list, list):
                    return len(' '.join(str(item) for item in body_list))
            return len(str(body))
        except:
            return len(str(body))

    def _categorize_primary_tag(self, tags: List[str]) -> str:
        """Data-driven categorization using actual tags."""
        if not tags:
            return 'Other'

        # Create categories based on most common patterns in Stack Overflow
        categories = {
            'Java': ['java', 'spring', 'android', 'kotlin', 'maven', 'gradle'],
            'Python': ['python', 'django', 'flask', 'pandas', 'numpy', 'matplotlib', 'scipy'],
            'JavaScript': ['javascript', 'js', 'node.js', 'nodejs', 'react', 'angular', 'vue', 'jquery'],
            'C#/.NET': ['c#', '.net', 'asp.net', 'winforms', 'wpf', 'entity-framework'],
            'C/C++': ['c++', 'c', 'gcc', 'visual-studio', 'cmake'],
            'Web Technologies': ['html', 'css', 'php', 'web', 'http', 'ajax', 'bootstrap'],
            'Database': ['sql', 'mysql', 'postgresql', 'database', 'sqlite', 'mongodb'],
            'Mobile': ['android', 'ios', 'mobile', 'swift', 'react-native', 'flutter'],
            'Data Science': ['machine-learning', 'data-analysis', 'statistics', 'r', 'tensorflow'],
            'DevOps/Tools': ['git', 'docker', 'linux', 'bash', 'jenkins', 'aws']
        }

        # Find first matching category
        for tag in tags:
            for category, category_tags in categories.items():
                if tag in category_tags:
                    return category

        # If no match found, return the first tag (capitalized)
        return tags[0].capitalize() if tags else 'Other'

    def _calculate_difficulty_level(self, row) -> str:
        """Calculate difficulty based on answer count and text length."""
        answer_count = row['answer_count']
        text_length = row['total_text_length']

        # Simple heuristic: more answers + longer text = higher difficulty
        if answer_count >= 5 and text_length > 1000:
            return 'High'
        elif answer_count >= 2 and text_length > 500:
            return 'Medium'
        else:
            return 'Low'

    def debug_answer_counting(self):
        """Debug the answer counting to understand the data structure."""
        print("\n🔍 DEBUGGING ANSWER COUNTING:")

        # Check answers CSV structure
        answers_df = pd.read_csv(self.answer_file)
        print(f"📊 Answers CSV columns: {answers_df.columns.tolist()}")
        print(f"📊 Answers CSV shape: {answers_df.shape}")
        print(f"📊 First 5 rows of answers:")
        print(answers_df.head())

        # Check questions CSV structure
        questions_df = pd.read_csv(self.question_file)
        print(f"\n📊 Questions CSV columns: {questions_df.columns.tolist()}")
        print(f"\n📊 Questions CSV shape: {questions_df.shape}")

        # Show sample answer_posts
        print(f"\n📄 Sample answer_posts from questions:")
        for i in range(5):
            qid = questions_df.iloc[i]['question_id']
            posts = questions_df.iloc[i]['answer_posts']
            print(f"   Question {qid}: {posts}")

        # Show current answer count distribution
        if hasattr(self, 'questions_df'):
            print(f"\n📈 Current answer count distribution:")
            answer_dist = self.questions_df['answer_count'].value_counts(
            ).sort_index()
            for count, freq in answer_dist.head(10).items():
                print(f"   {count} answers: {freq} questions")

    def create_overview_plot(self):
        """Create only the dataset overview plot."""
        print("📊 Creating dataset overview plot...")

        # Create main overview plot only
        self._create_main_overview()

        # Generate summary statistics
        self._generate_summary_table()

        print("✅ Dataset overview plot created successfully!")

    def _create_main_overview(self):
        """Create thesis-style overview with proper spacing and fonts."""
        # Use serif (Times New Roman) for plots to match thesis style
        plt.rcParams['font.family'] = 'serif'
        plt.rcParams['font.serif'] = [
            'Times New Roman', 'DejaVu Serif']

        fig = plt.figure(figsize=(16, 9))  # Wider figure for better spacing
        gs = fig.add_gridspec(2, 3, hspace=0.45, wspace=0.4,
                              left=0.08, right=0.96, top=0.85, bottom=0.15)

        # 1. Dataset Overview - Better spacing
        ax1 = fig.add_subplot(gs[0, 0])
        overview_data = {
            'Questions': len(self.questions_df),
            'Answers': len(self.answers_df),
            'Unique Tags': len(set(tag for tags in self.questions_df['tags_parsed'] for tag in tags)),
        }

        bars = ax1.bar(overview_data.keys(), overview_data.values(),
                       color=[colors_thesis['primary'],
                              colors_thesis['secondary'], colors_thesis['accent']],
                       alpha=0.85, edgecolor='white', linewidth=1.5)

        ax1.set_title('Dataset Overview', fontfamily='Times New Roman',
                      fontweight='bold', fontsize=13, pad=20)
        ax1.set_ylabel('Count', fontfamily='Times New Roman',
                       fontweight='bold', fontsize=11)
        ax1.grid(True, alpha=0.3, axis='y', color=colors_thesis['grid'])
        ax1.set_axisbelow(True)

        # Better positioned value labels with background for visibility
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + height*0.08,
                     f'{int(height):,}', ha='center', va='bottom',
                     fontfamily='Times New Roman', fontweight='bold', fontsize=10,
                     bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='none'))

        # Better tick label spacing - ensure GFS Didot everywhere
        ax1.tick_params(axis='both', which='major', labelsize=9, pad=5)
        for label in ax1.get_xticklabels():
            label.set_fontfamily('Times New Roman')
        for label in ax1.get_yticklabels():
            label.set_fontfamily('Times New Roman')

        # 2. Question Types - Clean pie chart with percentages only inside
        ax2 = fig.add_subplot(gs[0, 1])
        type_counts = self.questions_df['question_type_label'].value_counts()
        colors_pie = [colors_thesis['primary'],
                      colors_thesis['secondary'], colors_thesis['accent']]

        # Create pie chart with ONLY percentages inside (black font)
        wedges, texts, autotexts = ax2.pie(type_counts.values,
                                           labels=None,  # Remove external labels
                                           autopct='%1.1f%%',  # Only show percentages
                                           startangle=90,
                                           colors=colors_pie,
                                           wedgeprops=dict(
                                               edgecolor='white', linewidth=2),
                                           pctdistance=0.7,  # Move text closer to center
                                           textprops={'fontfamily': 'Times New Roman', 'fontweight': 'bold', 'fontsize': 11, 'color': 'black'})

        ax2.set_title('Question Types', fontfamily='Times New Roman',
                      fontweight='bold', fontsize=13, pad=20)

        # Create manual legend with Times New Roman font positioned further on the left
        legend_elements = [plt.Rectangle((0, 0), 1, 1, facecolor=colors_pie[i], edgecolor='white', linewidth=1)
                           for i in range(len(type_counts))]
        ax2.legend(legend_elements, type_counts.index,
                   loc='center left', bbox_to_anchor=(-0.8, 0.5),
                   prop={'family': 'Times New Roman', 'size': 10})

        # 3. Top Technologies - Horizontal bars with better spacing
        ax3 = fig.add_subplot(gs[0, 2])
        category_counts = self.questions_df['primary_tag_category'].value_counts().head(
            6)

        bars3 = ax3.barh(category_counts.index, category_counts.values,
                         color=colors_thesis['primary'], alpha=0.85,
                         edgecolor='white', linewidth=1, height=0.7)  # Thinner bars

        ax3.set_title('Top Technologies', fontfamily='Times New Roman',
                      fontweight='bold', fontsize=13, pad=20)
        ax3.set_xlabel('Questions', fontfamily='Times New Roman',
                       fontweight='bold', fontsize=11)
        ax3.grid(True, alpha=0.3, axis='x', color=colors_thesis['grid'])
        ax3.set_axisbelow(True)

        # Better tick spacing - ensure Times New Roman everywhere
        ax3.tick_params(axis='both', which='major', labelsize=9, pad=5)
        for label in ax3.get_yticklabels():
            label.set_fontfamily('Times New Roman')
        for label in ax3.get_xticklabels():
            label.set_fontfamily('Times New Roman')

        # Value labels with better positioning and background for visibility
        for bar in bars3:
            width = bar.get_width()
            ax3.text(width + width*0.05, bar.get_y() + bar.get_height()/2.,
                     f'{int(width)}', ha='left', va='center',
                     fontfamily='Times New Roman', fontweight='bold', fontsize=9,
                     bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8, edgecolor='none'))

        # 4. Answer Distribution - Much better spacing
        ax4 = fig.add_subplot(gs[1, :])
        answer_counts = self.questions_df['answer_count'].value_counts(
        ).sort_index()
        answer_counts_filtered = answer_counts[answer_counts.index <= 15]

        bars4 = ax4.bar(answer_counts_filtered.index, answer_counts_filtered.values,
                        color=colors_thesis['secondary'], alpha=0.85,
                        edgecolor='white', linewidth=1, width=0.8)

        ax4.set_title('Answer Distribution per Question', fontfamily='Times New Roman',
                      fontweight='bold', fontsize=13, pad=20)
        ax4.set_xlabel('Number of Answers', fontfamily='Times New Roman',
                       fontweight='bold', fontsize=11)
        ax4.set_ylabel('Number of Questions', fontfamily='Times New Roman',
                       fontweight='bold', fontsize=11)
        ax4.grid(True, alpha=0.3, axis='y', color=colors_thesis['grid'])
        ax4.set_axisbelow(True)

        # Better tick spacing and rotation - ensure Times New Roman everywhere
        ax4.tick_params(axis='both', which='major', labelsize=9, pad=5)
        for label in ax4.get_xticklabels():
            label.set_fontfamily('Times New Roman')
        for label in ax4.get_yticklabels():
            label.set_fontfamily('Times New Roman')

        # Value labels for all bars with improved visibility
        for bar in bars4:
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + height*0.05,
                     f'{int(height)}', ha='center', va='bottom',
                     fontfamily='Times New Roman', fontweight='bold', fontsize=8,
                     bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.9, edgecolor='none'))

        # Main title with better positioning
        fig.suptitle('SOSum Dataset: Overview Statistics',
                     fontfamily='Times New Roman', fontsize=16, fontweight='bold',
                     y=0.94, color=colors_thesis['text'])

        plt.savefig(self.plots_dir / 'dataset_overview.png', dpi=300, bbox_inches='tight',
                    facecolor='white', edgecolor='none', pad_inches=0.2)  # Extra padding
        plt.show()

    def _generate_summary_table(self):
        """Generate summary statistics for the report."""
        print("\n" + "="*70)
        print("📊 SOSUM DATASET SUMMARY FOR THESIS REPORT")
        print("="*70)

        # Calculate key metrics
        total_questions = len(self.questions_df)
        total_answers = len(self.answers_df)
        unique_tags = len(
            set(tag for tags in self.questions_df['tags_parsed'] for tag in tags))

        print(f"\n📈 DATASET OVERVIEW:")
        print(f"   • Total Questions: {total_questions:,}")
        print(f"   • Total Answers: {total_answers:,}")
        print(
            f"   • Questions with Answers: {(self.questions_df['answer_count'] > 0).sum():,} ({(self.questions_df['answer_count'] > 0).mean()*100:.1f}%)")
        print(f"   • Unique Tags: {unique_tags:,}")
        print(
            f"   • Average Answers per Question: {self.questions_df['answer_count'].mean():.2f}")

        print(f"\n🏷️  QUESTION TYPES:")
        for qtype, count in self.questions_df['question_type_label'].value_counts().items():
            percentage = (count / total_questions) * 100
            print(f"   • {qtype}: {count:,} ({percentage:.1f}%)")

        print(f"\n💻 TOP TECHNOLOGIES:")
        for tech, count in self.questions_df['primary_tag_category'].value_counts().head(5).items():
            percentage = (count / total_questions) * 100
            print(f"   • {tech}: {count:,} ({percentage:.1f}%)")

        # Save summary for thesis
        summary = {
            'overview': {
                'total_questions': total_questions,
                'total_answers': total_answers,
                'unique_tags': unique_tags,
                'avg_answers_per_question': round(self.questions_df['answer_count'].mean(), 2)
            },
            'question_types': self.questions_df['question_type_label'].value_counts().to_dict(),
            'top_technologies': self.questions_df['primary_tag_category'].value_counts().head(10).to_dict()
        }

        with open(self.plots_dir.parent / 'dataset_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)

        print(
            f"\n✅ Summary saved to: {self.plots_dir.parent / 'dataset_summary.json'}")
        print("="*70)


def main():
    """Run complete analysis for thesis report."""
    analyzer = SOSUMDatasetAnalyzer('datasets/sosum/')

    # First debug to see the data structure
    analyzer.debug_answer_counting()

    # Then create overview plot
    analyzer.create_overview_plot()


if __name__ == "__main__":
    main()
