"""Domain-specific prefixes for instruction and UI-element embeddings."""

import re

INSTRUCTION_TEMPLATES_COMBINED = {
    # Scientific computing
    ("matlab", "macos"): "Represent the MATLAB macOS element (plot toolbar, variable inspector, command window, function browser, toolstrip) for semantic matching with user instruction:",
    ("matlab", "windows"): "Represent the MATLAB Windows element (ribbon tool, workspace panel, command prompt, function editor, app designer) for semantic matching with user instruction:",
    ("origin", "windows"): "Represent the Origin Windows element (graph toolbar, worksheet column, fitting dialog, statistics panel, analysis menu) for semantic matching with user instruction:",
    ("stata", "windows"): "Represent the Stata Windows element (data editor cell, command syntax, result viewer, graph editor, variable manager) for semantic matching with user instruction:",
    ("eviews", "windows"): "Represent the EViews Windows element (workfile object, equation toolbar, forecasting dialog, series operation, model specification) for semantic matching with user instruction:",

    # CAD and 3D design
    ("solidworks", "windows"): "Represent the SolidWorks Windows element (feature tree, sketch tool, assembly mate, drawing dimension, design library) for semantic matching with user instruction:",
    ("autocad", "windows"): "Represent the AutoCAD Windows element (ribbon panel, command line, layer control, dimension style, block library) for semantic matching with user instruction:",
    ("inventor", "windows"): "Represent the Inventor Windows element (browser panel, sketch constraint, assembly tool, drawing view, simulation setup) for semantic matching with user instruction:",
    ("blender", "windows"): "Represent the Blender Windows element (outliner panel, modifier stack, shader editor, timeline keyframe, render property) for semantic matching with user instruction:",
    ("blender", "macos"): "Represent the Blender macOS element (outliner hierarchy, modifier panel, node editor, animation timeline, render setting) for semantic matching with user instruction:",
    ("blender", "linux"): "Represent the Blender Linux element (scene collection, geometry node, material editor, timeline marker, compositor node) for semantic matching with user instruction:",
    ("unreal_engine", "windows"): "Represent the Unreal Engine Windows element (content browser, blueprint graph, level viewport, details panel, world outliner) for semantic matching with user instruction:",

    # Electronic design automation
    ("vivado", "windows"): "Represent the Vivado Windows element (project navigator, synthesis setting, implementation tool, IP catalog, timing constraint) for semantic matching with user instruction:",
    ("vivado", "linux"): "Represent the Vivado Linux element (flow navigator, synthesis option, place & route, IP integrator, XDC constraint) for semantic matching with user instruction:",
    ("quartus", "windows"): "Represent the Quartus Prime Windows element (project navigator, compilation flow, pin planner, timing analyzer, signal tap) for semantic matching with user instruction:",

    # Multimedia
    ("premiere", "windows"): "Represent the Premiere Pro Windows element (timeline panel, effect control, sequence setting, export queue, media browser) for semantic matching with user instruction:",
    ("premiere", "macos"): "Represent the Premiere Pro macOS element (timeline track, effect panel, sequence preset, export dialog, source monitor) for semantic matching with user instruction:",
    ("davinci", "macos"): "Represent the DaVinci Resolve macOS element (color wheel, fusion page, fairlight mixer, timeline tool, deliver page) for semantic matching with user instruction:",
    ("davinci", "windows"): "Represent the DaVinci Resolve Windows element (primary grading, node editor, audio mixer, edit timeline, render setting) for semantic matching with user instruction:",
    ("fruitloops", "windows"): "Represent the FL Studio Windows element (channel rack, mixer track, piano roll, playlist, plugin picker) for semantic matching with user instruction:",
    ("photoshop", "windows"): "Represent the Photoshop Windows element (layer panel, adjustment layer, filter gallery, brush preset, selection tool) for semantic matching with user instruction:",
    ("photoshop", "macos"): "Represent the Photoshop macOS element (layer stack, adjustment panel, filter menu, tool preset, selection option) for semantic matching with user instruction:",
    ("illustrator", "windows"): "Represent the Illustrator Windows element (artboard panel, path tool, stroke option, gradient editor, symbol library) for semantic matching with user instruction:",
    ("illustrator", "macos"): "Represent the Illustrator macOS element (artboard tool, pen tool, appearance panel, gradient tool, symbol panel) for semantic matching with user instruction:",

    # Office
    ("word", "macos"): "Represent the Word macOS element (ribbon tab, style gallery, review tool, reference manager, format inspector) for semantic matching with user instruction:",
    ("word", "windows"): "Represent the Word Windows element (ribbon group, style pane, track changes, citation tool, page layout) for semantic matching with user instruction:",
    ("excel", "macos"): "Represent the Excel macOS element (ribbon tool, formula bar, pivot table, chart option, data validation) for semantic matching with user instruction:",
    ("excel", "windows"): "Represent the Excel Windows element (ribbon command, function wizard, pivot chart, conditional format, data tool) for semantic matching with user instruction:",
    ("powerpoint", "windows"): "Represent the PowerPoint Windows element (slide layout, animation pane, transition gallery, design theme, master slide) for semantic matching with user instruction:",
    ("powerpoint", "macos"): "Represent the PowerPoint macOS element (slide template, animation option, transition effect, theme variant, slide master) for semantic matching with user instruction:",

    # Developer tools
    ("vscode", "macos"): "Represent the VS Code macOS element (command palette, sidebar panel, debug console, terminal tab, extension view) for semantic matching with user instruction:",
    ("vscode", "windows"): "Represent the VS Code Windows element (command menu, explorer panel, debug toolbar, integrated terminal, extension manager) for semantic matching with user instruction:",
    ("vscode", "linux"): "Represent the VS Code Linux element (command search, file explorer, debugger panel, terminal shell, plugin marketplace) for semantic matching with user instruction:",
    ("pycharm", "macos"): "Represent the PyCharm macOS element (project tool window, run configuration, debugger tab, python console, structure view) for semantic matching with user instruction:",
    ("pycharm", "windows"): "Represent the PyCharm Windows element (project explorer, run menu, debug panel, python interpreter, code structure) for semantic matching with user instruction:",
    ("android_studio", "macos"): "Represent the Android Studio macOS element (project structure, layout editor, logcat panel, build variant, device manager) for semantic matching with user instruction:",
    ("android_studio", "windows"): "Represent the Android Studio Windows element (project view, XML editor, logcat window, gradle tool, AVD manager) for semantic matching with user instruction:",

    # Virtualization
    ("vmware", "macos"): "Represent the VMware Fusion macOS element (virtual machine library, settings panel, snapshot manager, network adapter, shared folder) for semantic matching with user instruction:",
    ("vmware", "windows"): "Represent the VMware Workstation Windows element (VM library, hardware setting, snapshot tool, network editor, shared folder) for semantic matching with user instruction:",
}

INSTRUCTION_TEMPLATES_APP = {
    "matlab": "Represent the MATLAB workspace element (plot tool, variable editor, command window, function, toolbox) for semantic matching with user instruction:",
    "origin": "Represent the Origin data analysis element (graph tool, worksheet operation, fitting function, statistics panel) for semantic matching with user instruction:",
    "stata": "Represent the Stata statistical element (data editor, command syntax, result window, graph editor) for semantic matching with user instruction:",
    "eviews": "Represent the EViews econometric element (workfile object, equation estimation, forecasting tool, series operation) for semantic matching with user instruction:",
    "solidworks": "Represent the SolidWorks CAD element (sketch tool, feature command, assembly mate, drawing dimension) for semantic matching with user instruction:",
    "autocad": "Represent the AutoCAD drafting element (draw command, modify tool, layer control, dimension style) for semantic matching with user instruction:",
    "inventor": "Represent the Inventor CAD element (part modeling, assembly constraint, drawing view, simulation tool) for semantic matching with user instruction:",
    "blender": "Represent the Blender 3D element (modifier, shader node, animation keyframe, sculpting brush, render setting) for semantic matching with user instruction:",
    "unreal_engine": "Represent the Unreal Engine element (blueprint node, material editor, level viewport, actor component) for semantic matching with user instruction:",
    "vivado": "Represent the Vivado FPGA element (synthesis option, implementation tool, IP catalog, timing constraint) for semantic matching with user instruction:",
    "quartus": "Represent the Quartus Prime element (compilation setting, pin planner, timing analyzer, signal tap) for semantic matching with user instruction:",
    "premiere": "Represent the Premiere Pro element (timeline clip, effect control, sequence setting, export preset) for semantic matching with user instruction:",
    "davinci": "Represent the DaVinci Resolve element (color grading wheel, fusion node, fairlight mixer, timeline tool) for semantic matching with user instruction:",
    "fruitloops": "Represent the FL Studio element (channel rack, mixer track, piano roll, plugin effect) for semantic matching with user instruction:",
    "photoshop": "Represent the Photoshop element (layer panel, adjustment tool, filter effect, brush preset, selection tool) for semantic matching with user instruction:",
    "illustrator": "Represent the Illustrator element (vector path, artboard, stroke panel, gradient tool, symbol library) for semantic matching with user instruction:",
    "word": "Represent the Word document element (formatting ribbon, style gallery, review tool, reference manager) for semantic matching with user instruction:",
    "excel": "Represent the Excel spreadsheet element (formula bar, pivot table, chart tool, data validation, conditional format) for semantic matching with user instruction:",
    "powerpoint": "Represent the PowerPoint slide element (slide layout, animation pane, transition effect, design theme) for semantic matching with user instruction:",
    "vscode": "Represent the VS Code editor element (command palette, extension, debug panel, terminal, sidebar) for semantic matching with user instruction:",
    "pycharm": "Represent the PyCharm IDE element (refactor tool, run configuration, debugger, project structure, code inspection) for semantic matching with user instruction:",
    "android_studio": "Represent the Android Studio IDE element (layout editor, emulator control, gradle tool, logcat, build variant) for semantic matching with user instruction:",
    "vmware": "Represent the VMware element (virtual machine control, network adapter, snapshot manager, settings panel) for semantic matching with user instruction:",
    "gitlab": "Represent the GitLab interface element (repository tree, merge request, CI/CD pipeline, issue board) for semantic matching with user instruction:",
    "github": "Represent the GitHub interface element (repo file, pull request, action workflow, project board) for semantic matching with user instruction:",
    "jira": "Represent the Jira project element (issue card, sprint board, workflow transition, backlog item) for semantic matching with user instruction:",
}

INSTRUCTION_TEMPLATES_PLATFORM = {
    "macos": "Represent the macOS application element (menu bar item, toolbar button, preference pane, dock icon) for semantic matching with user instruction:",
    "windows": "Represent the Windows application element (ribbon control, context menu, taskbar tool, system tray) for semantic matching with user instruction:",
    "linux": "Represent the Linux application element (menu item, toolbar action, terminal command, system setting) for semantic matching with user instruction:",
    "web": "Represent the web page element (navigation link, form input, action button, content section) for semantic matching with user instruction:",
    "ios": "Represent the iOS app element (navigation bar, tab bar, gesture area, action sheet) for semantic matching with user instruction:",
    "android": "Represent the Android app element (action bar, floating button, drawer menu, bottom sheet) for semantic matching with user instruction:",
}

DEFAULT_INSTRUCTION = "Represent the GUI element for semantic matching with user instruction:"


APP_ALIASES = {
    'unreal': 'unreal_engine', 'ue4': 'unreal_engine', 'ue5': 'unreal_engine',
    'premiere_pro': 'premiere', 'davinci_resolve': 'davinci',
    'fl_studio': 'fruitloops', 'flstudio': 'fruitloops',
    'vmware_workstation': 'vmware', 'vmware_fusion': 'vmware',
    'quartus_prime': 'quartus', 'ppt': 'powerpoint',
    'adobe_photoshop': 'photoshop', 'adobe_illustrator': 'illustrator',
    'ms_word': 'word', 'microsoft_word': 'word',
    'ms_excel': 'excel', 'microsoft_excel': 'excel',
    'ms_powerpoint': 'powerpoint', 'microsoft_powerpoint': 'powerpoint',
    'visual_studio_code': 'vscode', 'code': 'vscode',
}

PLATFORM_ALIASES = {
    'win32': 'windows', 'win': 'windows',
    'darwin': 'macos', 'osx': 'macos', 'mac': 'macos',
    'ubuntu': 'linux', 'debian': 'linux', 'fedora': 'linux',
    'mobile': 'android', 'mobile_web': 'web',
    'ipad_os': 'ios', 'ipados': 'ios',
}

def _canon(s: str) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"[\s/\-\.]+", "_", s)
    return s

def _canon_app(app: str) -> str:
    app = _canon(app)
    return APP_ALIASES.get(app, app)

def _canon_platform(p: str) -> str:
    p = _canon(p)
    return PLATFORM_ALIASES.get(p, p)


def select_instruction(row):
    """Select the shared embedding prefix from application and platform."""
    raw_app = row.get("application", "")
    raw_plat = row.get("platform", row.get("data_source", ""))
    app = _canon_app(raw_app)
    plat = _canon_platform(raw_plat)

    key = (app, plat)
    if key in INSTRUCTION_TEMPLATES_COMBINED:
        return INSTRUCTION_TEMPLATES_COMBINED[key]
    if app in INSTRUCTION_TEMPLATES_APP:
        return INSTRUCTION_TEMPLATES_APP[app]
    if app:
        for app_key, prefix in INSTRUCTION_TEMPLATES_APP.items():
            if app_key in app or app in app_key:
                return prefix
    if plat in INSTRUCTION_TEMPLATES_PLATFORM:
        return INSTRUCTION_TEMPLATES_PLATFORM[plat]
    if plat:
        for platform_key, prefix in INSTRUCTION_TEMPLATES_PLATFORM.items():
            if platform_key in plat or plat in platform_key:
                return prefix
    return DEFAULT_INSTRUCTION
