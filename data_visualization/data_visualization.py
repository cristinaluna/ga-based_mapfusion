import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from matplotlib.backends.backend_pdf import PdfPages

class GeneticAlgorithmReport:
    def __init__(self, results_path: str, overlap_path: str):
        """
        Initialize with paths to CSV files.
        """
        self.results_path = results_path
        self.overlap_path = overlap_path
        self.results_df = None
        self.overlap_df = None
        self.merged_df = None

        sns.set_theme(style="whitegrid")
        plt.rcParams["figure.figsize"] = (10, 6)

    def load_data(self):
        """
        Load data and merge on map_number/index.
        """
        self.results_df = pd.read_csv(self.results_path)
        self.overlap_df = pd.read_csv(self.overlap_path)
        self.merged_df = self.results_df.merge(self.overlap_df, left_on="index", right_on="map_number")

        self.merged_df.to_csv("merged_data.csv", index=False)


    def _save_plot(self, filename, plot_func):
        """
        Run a plot function and save to filename.
        """
        plt.figure()
        plot_func()
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()

    def generate_report(self, save_path="report_output"):
        """
        Generate all plots and export as PNGs and PDF.
        """
        self.load_data()
        os.makedirs(save_path, exist_ok=True)
        pdf_path = os.path.join(save_path, "report.pdf")

        with PdfPages(pdf_path) as pdf:
            # 1. Histogram of Final Score
            def p1():
                sns.histplot(self.results_df["final_score"], bins=20, kde=True)
                plt.title("Distribution of Final Score")
                plt.xlabel("Final Score")
                plt.ylabel("Frequency")
            # self._save_plot(f"{save_path}/01_final_score_hist.png", p1)
            p1(); pdf.savefig(); plt.close()

            # 2. Final Score vs Generations
            def p2():
                sns.scatterplot(data=self.results_df, x="num_generations", y="final_score")
                plt.title("Final Score vs Number of Generations")
                plt.xlabel("Number of Generations")
                plt.ylabel("Final Score")
            # self._save_plot(f"{save_path}/02_score_vs_generations.png", p2)
            p2(); pdf.savefig(); plt.close()

            # 3. Boxplot of stats
            def p3():
                sns.boxplot(data=self.results_df[["max", "mean", "median"]])
                plt.title("Distribution of Population Statistics")
                plt.ylabel("Value")
            # self._save_plot(f"{save_path}/03_population_stats_box.png", p3)
            p3(); pdf.savefig(); plt.close()

            # 4. Success comparison
            def p4():
                sns.boxplot(data=self.results_df, x="successful", y="final_score")
                plt.title("Final Score by Success")
                plt.xlabel("Run Successful (Y/N)")
                plt.ylabel("Final Score")
            # self._save_plot(f"{save_path}/04_success_comparison.png", p4)
            p4(); pdf.savefig(); plt.close()

            # 5. Correlation heatmap
            def p5():
                corr = self.results_df[["max", "mean", "median", "num_generations", "final_score"]].corr()
                sns.heatmap(corr, annot=True, cmap="coolwarm")
                plt.title("Correlation Matrix")
            # self._save_plot(f"{save_path}/05_correlation_heatmap.png", p5)
            p5(); pdf.savefig(); plt.close()

            # 6. Overlap distribution
            def p6():
                sns.histplot(self.overlap_df["overlap"], bins=20, kde=True)
                plt.title("Distribution of Overlap")
                plt.xlabel("Overlap")
                plt.ylabel("Frequency")
            # self._save_plot(f"{save_path}/06_overlap_distribution.png", p6)
            p6(); pdf.savefig(); plt.close()

            # 7. Overlap vs Final Score scatter
            def p7():
                sns.scatterplot(data=self.merged_df, x="overlap", y="final_score")
                plt.title("Overlap vs Final Score")
                plt.xlabel("Overlap")
                plt.ylabel("Final Score")
            # self._save_plot(f"{save_path}/07_overlap_vs_score_scatter.png", p7)
            p7(); pdf.savefig(); plt.close()

            # 8. Overlap vs Final Score regression
            def p8():
                sns.lmplot(data=self.merged_df, x="overlap", y="final_score", ci=95, line_kws={"color": "red"})
                plt.title("Linear Regression: Overlap vs Final Score")
                plt.xlabel("Overlap")
                plt.ylabel("Final Score")
            # sns_plot = sns.lmplot(data=self.merged_df, x="overlap", y="final_score", ci=95, line_kws={"color": "red"})
            # sns_plot.savefig(f"{save_path}/08_overlap_vs_score_regression.png")
            p8(); pdf.savefig(); plt.close()


            # 9. Pairplot
            def p9():
                pairplot = sns.pairplot(self.merged_df, vars=["overlap", "max", "mean", "median", "final_score"])
                pairplot.figure.suptitle("Relations between Overlap and Metrics", y=1.003)
            p9(); pdf.savefig(); plt.close()

            def p10():
                self.merged_df["map_id"] = self.merged_df["map_number"].astype(str)

                plt.figure(figsize=(20, 6))

                sns.scatterplot(
                    data=self.merged_df,
                    x="map_id",
                    y="final_score",
                    size="overlap",
                    hue="successful",
                    palette={"Y": "green", "N": "red"},
                    alpha=0.7,
                    sizes=(50, 200),
                    edgecolor="gray",
                    linewidth=0.5
                )

                plt.title("Final Score by Map, Overlap, and Success")
                plt.xlabel("Map Number")
                plt.ylabel("Final Score")
                plt.xticks(rotation=90, ha='center')
                plt.legend(title="Successful and Overlap", bbox_to_anchor=(1.05, 1), loc='upper left')
                plt.tight_layout()

            p10(); pdf.savefig(); plt.close()



        print(f"Report saved in '{save_path}' as a PDF.")
