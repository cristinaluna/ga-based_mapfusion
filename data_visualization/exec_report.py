from data_visualization import GeneticAlgorithmReport

report = GeneticAlgorithmReport(
    results_path="ga_fusion_syntheticexperiments_data/fusion_summary.csv",
    overlap_path="map_overlaps.csv"
)
report.generate_report(save_path="report_output")

