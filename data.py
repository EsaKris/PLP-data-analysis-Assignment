import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
import os
import warnings
warnings.filterwarnings('ignore')

class DataAnalysisApp:
    def __init__(self):
        self.data = None
        self.loaded = False
        self.visualizations = []
        
    def load_dataset(self, filepath=None):
        """Load dataset from file or use default iris dataset"""
        try:
            if filepath and os.path.exists(filepath):
                self.data = pd.read_csv(filepath)
                print(f"Dataset loaded successfully from {filepath}")
            else:
                # Load default iris dataset if no file provided or file not found
                iris = load_iris()
                self.data = pd.DataFrame(data=iris.data, columns=iris.feature_names)
                self.data['species'] = iris.target
                self.data['species'] = self.data['species'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})
                print("Default Iris dataset loaded")
            
            self.loaded = True
            self.display_head()
            self.check_data_quality()
            
        except Exception as e:
            print(f"Error loading dataset: {str(e)}")
            self.loaded = False
    
    def display_head(self, n=5):
        """Display first n rows of the dataset"""
        if self.loaded:
            print("\nFirst few rows of the dataset:")
            print(self.data.head(n))
        else:
            print("No dataset loaded. Please load a dataset first.")
    
    def check_data_quality(self):
        """Check data types and missing values"""
        if self.loaded:
            print("\nData Quality Check:")
            print("\nData Types:")
            print(self.data.dtypes)
            
            print("\nMissing Values:")
            print(self.data.isnull().sum())
            
            # Clean data by dropping rows with missing values
            initial_rows = len(self.data)
            self.data.dropna(inplace=True)
            cleaned_rows = len(self.data)
            
            if initial_rows != cleaned_rows:
                print(f"\nDropped {initial_rows - cleaned_rows} rows with missing values.")
            else:
                print("\nNo missing values found in the dataset.")
        else:
            print("No dataset loaded. Please load a dataset first.")
    
    def basic_statistics(self):
        """Compute basic statistics for numerical columns"""
        if self.loaded:
            print("\nBasic Statistics for Numerical Columns:")
            print(self.data.describe())
        else:
            print("No dataset loaded. Please load a dataset first.")
    
    def group_analysis(self, group_col, agg_col):
        """Perform grouping and aggregation"""
        if self.loaded:
            try:
                print(f"\nGroup Analysis: Mean of {agg_col} by {group_col}")
                grouped = self.data.groupby(group_col)[agg_col].mean()
                print(grouped)
                
                # Identify patterns
                max_val = grouped.idxmax()
                min_val = grouped.idxmin()
                print(f"\nPatterns: Highest average {agg_col} in {max_val}, lowest in {min_val}")
                
                return grouped
            except KeyError as e:
                print(f"Column not found: {str(e)}")
            except Exception as e:
                print(f"Error in group analysis: {str(e)}")
        else:
            print("No dataset loaded. Please load a dataset first.")
    
    def create_line_chart(self, x_col, y_col, title="Line Chart"):
        """Create a line chart showing trends"""
        if self.loaded:
            try:
                plt.figure(figsize=(10, 6))
                if x_col in self.data.columns and y_col in self.data.columns:
                    # Sort by x_col for proper line plotting
                    temp_df = self.data.sort_values(x_col)
                    plt.plot(temp_df[x_col], temp_df[y_col], marker='o')
                    plt.title(title)
                    plt.xlabel(x_col)
                    plt.ylabel(y_col)
                    plt.grid(True)
                    plt.tight_layout()
                    
                    # Save the visualization reference
                    self.visualizations.append(("Line Chart", plt.gcf()))
                    plt.show()
                else:
                    print("One or both specified columns not found in dataset.")
            except Exception as e:
                print(f"Error creating line chart: {str(e)}")
        else:
            print("No dataset loaded. Please load a dataset first.")
    
    def create_bar_chart(self, x_col, y_col, title="Bar Chart"):
        """Create a bar chart comparing values across categories"""
        if self.loaded:
            try:
                plt.figure(figsize=(10, 6))
                sns.barplot(x=x_col, y=y_col, data=self.data, ci=None)
                plt.title(title)
                plt.xlabel(x_col)
                plt.ylabel(y_col)
                plt.xticks(rotation=45)
                plt.tight_layout()
                
                # Save the visualization reference
                self.visualizations.append(("Bar Chart", plt.gcf()))
                plt.show()
            except Exception as e:
                print(f"Error creating bar chart: {str(e)}")
        else:
            print("No dataset loaded. Please load a dataset first.")
    
    def create_histogram(self, col, bins=10, title="Histogram"):
        """Create a histogram of a numerical column"""
        if self.loaded:
            try:
                plt.figure(figsize=(10, 6))
                sns.histplot(self.data[col], bins=bins, kde=True)
                plt.title(title)
                plt.xlabel(col)
                plt.ylabel("Frequency")
                plt.tight_layout()
                
                # Save the visualization reference
                self.visualizations.append(("Histogram", plt.gcf()))
                plt.show()
            except Exception as e:
                print(f"Error creating histogram: {str(e)}")
        else:
            print("No dataset loaded. Please load a dataset first.")
    
    def create_scatter_plot(self, x_col, y_col, hue_col=None, title="Scatter Plot"):
        """Create a scatter plot between two numerical columns"""
        if self.loaded:
            try:
                plt.figure(figsize=(10, 6))
                if hue_col and hue_col in self.data.columns:
                    sns.scatterplot(x=x_col, y=y_col, hue=hue_col, data=self.data)
                else:
                    sns.scatterplot(x=x_col, y=y_col, data=self.data)
                plt.title(title)
                plt.xlabel(x_col)
                plt.ylabel(y_col)
                plt.tight_layout()
                
                # Save the visualization reference
                self.visualizations.append(("Scatter Plot", plt.gcf()))
                plt.show()
            except Exception as e:
                print(f"Error creating scatter plot: {str(e)}")
        else:
            print("No dataset loaded. Please load a dataset first.")
    
    def show_visualizations(self):
        """Display all created visualizations"""
        if not self.visualizations:
            print("No visualizations created yet.")
            return
            
        for name, fig in self.visualizations:
            print(f"\nDisplaying {name}")
            plt.figure(fig.number)
            plt.show()
    
    def run_demo(self):
        """Run a demo analysis with the default dataset"""
        print("\nRunning Demo Analysis with Iris Dataset...")
        
        # Load default dataset
        self.load_dataset()
        
        # Basic statistics
        self.basic_statistics()
        
        # Group analysis
        self.group_analysis('species', 'sepal length (cm)')
        
        # Create visualizations
        self.create_line_chart('sepal length (cm)', 'petal length (cm)', 'Sepal vs Petal Length Trend')
        self.create_bar_chart('species', 'sepal width (cm)', 'Average Sepal Width by Species')
        self.create_histogram('petal length (cm)', bins=15, title='Distribution of Petal Lengths')
        self.create_scatter_plot('sepal length (cm)', 'petal length (cm)', 'species', 'Sepal vs Petal Length by Species')
        
        print("\nDemo completed! Check out the visualizations.")

def main():
    app = DataAnalysisApp()
    
    print("""
    Data Analysis and Visualization Application
    ------------------------------------------
    
    This application allows you to:
    1. Load and explore datasets
    2. Perform basic data analysis
    3. Create various visualizations
    
    The application comes with the Iris dataset as default.
    You can also load your own CSV file.
    """)
    
    while True:
        print("\nMenu:")
        print("1. Load dataset (default: Iris dataset)")
        print("2. Display dataset head")
        print("3. Check data quality")
        print("4. Show basic statistics")
        print("5. Perform group analysis")
        print("6. Create line chart")
        print("7. Create bar chart")
        print("8. Create histogram")
        print("9. Create scatter plot")
        print("10. Show all visualizations")
        print("11. Run demo analysis")
        print("12. Exit")
        
        choice = input("Enter your choice (1-12): ")
        
        if choice == '1':
            filepath = input("Enter CSV file path (leave blank for Iris dataset): ").strip()
            app.load_dataset(filepath if filepath else None)
        
        elif choice == '2':
            n = input("Number of rows to display (default 5): ").strip()
            app.display_head(int(n) if n.isdigit() else 5)
        
        elif choice == '3':
            app.check_data_quality()
        
        elif choice == '4':
            app.basic_statistics()
        
        elif choice == '5':
            if app.loaded:
                group_col = input("Enter column to group by: ")
                agg_col = input("Enter column to calculate mean: ")
                app.group_analysis(group_col, agg_col)
            else:
                print("No dataset loaded. Please load a dataset first.")
        
        elif choice == '6':
            if app.loaded:
                x_col = input("Enter x-axis column: ")
                y_col = input("Enter y-axis column: ")
                title = input("Enter chart title (optional): ")
                app.create_line_chart(x_col, y_col, title if title else f"{y_col} over {x_col}")
            else:
                print("No dataset loaded. Please load a dataset first.")
        
        elif choice == '7':
            if app.loaded:
                x_col = input("Enter categorical column: ")
                y_col = input("Enter numerical column: ")
                title = input("Enter chart title (optional): ")
                app.create_bar_chart(x_col, y_col, title if title else f"{y_col} by {x_col}")
            else:
                print("No dataset loaded. Please load a dataset first.")
        
        elif choice == '8':
            if app.loaded:
                col = input("Enter numerical column: ")
                bins = input("Enter number of bins (default 10): ")
                title = input("Enter chart title (optional): ")
                app.create_histogram(col, int(bins) if bins.isdigit() else 10, 
                                    title if title else f"Distribution of {col}")
            else:
                print("No dataset loaded. Please load a dataset first.")
        
        elif choice == '9':
            if app.loaded:
                x_col = input("Enter x-axis column: ")
                y_col = input("Enter y-axis column: ")
                hue_col = input("Enter column for color grouping (optional): ")
                title = input("Enter chart title (optional): ")
                app.create_scatter_plot(x_col, y_col, hue_col if hue_col else None, 
                                      title if title else f"{y_col} vs {x_col}")
            else:
                print("No dataset loaded. Please load a dataset first.")
        
        elif choice == '10':
            app.show_visualizations()
        
        elif choice == '11':
            app.run_demo()
        
        elif choice == '12':
            print("Exiting the application. Goodbye!")
            break
        
        else:
            print("Invalid choice. Please enter a number between 1 and 12.")

if __name__ == "__main__":
    main()