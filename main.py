from chart import draw_line_chart, draw_heatmap, data_creation, model_creation_dtr, visualize_data_tree_prediction, \
    model_creator_linear_regression, visualize_data_linear_regression_prediction


if __name__ == '__main__':
    draw_line_chart()
    draw_heatmap()
    df = data_creation()
    model_creation_dtr()
    visualize_data_tree_prediction()
    model_creator_linear_regression()
    visualize_data_linear_regression_prediction()
