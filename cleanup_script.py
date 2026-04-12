import sys
import ast

with open('src/facemesh_app/ellipse_estimator.py', 'r', encoding='utf-8') as f:
    source = f.read()

# Define the functions to remove
funcs_to_remove = [
    '_adaptive_canny_by_percent',
    '_sample_ellipse_support',
    '_dir_index',
    '_edge_segments_cpel',
    '_thin_segments',
    '_segment_directions',
    '_lb_lsd',
    '_split_arcs',
    '_merge_arcs',
    '_ellipse_support_ratio',
    '_ellipse_angle_coverage_deg',
    '_ellipse_norm_val_at_point',
    '_arc_fit_quality',
    '_ellipse_contour_match_stats',
    '_five_point_ellipse_candidates'
]

# We also need to remove _DIR_OFFSETS variable
class FunctionRemover(ast.NodeVisitor):
    def __init__(self):
        self.remove_nodes = []
    
    def visit_FunctionDef(self, node):
        if node.name in funcs_to_remove:
            self.remove_nodes.append(node)
            
    def visit_Assign(self, node):
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id == '_DIR_OFFSETS':
                self.remove_nodes.append(node)

tree = ast.parse(source)
visitor = FunctionRemover()
visitor.visit(tree)

# Build a list of lines to remove
lines_to_remove = set()
for node in visitor.remove_nodes:
    for line in range(node.lineno, node.end_lineno + 1):
        lines_to_remove.add(line)

new_lines = []
lines = source.split('\n')
for i, line in enumerate(lines):
    if (i + 1) not in lines_to_remove:
        new_lines.append(line)

new_source = '\n'.join(new_lines)

# Rename things
new_source = new_source.replace('def arc_method_ellipse(', 'def binarization_ellipse(')
new_source = new_source.replace('arc_ellipse = arc_method_ellipse(', 'bin_ellipse = binarization_ellipse(')
new_source = new_source.replace('if arc_ellipse is None:', 'if bin_ellipse is None:')
new_source = new_source.replace('"reason": "no arc ellipse"', '"reason": "no binarization ellipse"')
new_source = new_source.replace('ellipse = arc_ellipse', 'ellipse = bin_ellipse')
new_source = new_source.replace('"method": "arc_support_lsd"', '"method": "binarization"')
new_source = new_source.replace('ellipse_source = "arcSupportLSD"', 'ellipse_source = "binarization"')
new_source = new_source.replace('cv2.ellipse(arc_vis, arc_ellipse,', 'cv2.ellipse(arc_vis, bin_ellipse,')
new_source = new_source.replace('"arc_08_best_fit_bgr"', '"bin_08_best_fit_bgr"')
new_source = new_source.replace('"arc_09_best_fit"', '"bin_09_best_fit"')

# Ensure no hanging empty lines
import re
new_source = re.sub(r'\n{4,}', '\n\n', new_source)

with open('src/facemesh_app/ellipse_estimator.py', 'w', encoding='utf-8') as f:
    f.write(new_source)

print("Done cleaning and renaming!")