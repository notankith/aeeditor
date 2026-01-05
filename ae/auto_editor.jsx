(function () {
    app.beginUndoGroup("Auto Editor");

    alert("Processing... Please wait.");

    // locate the python controller relative to this JSX file
    var jsxFile = new File($.fileName);
    var projectRoot = new Folder(jsxFile.parent.parent.fsName);
    var pyController = new File(projectRoot.fsName + "/python/controller.py");

    // ask user for voiceover file
    var selectedVoiceover = File.openDialog("Select voiceover audio (WAV/MP3)");
    if (selectedVoiceover == null) {
        alert("No audio selected. Canceling.");
        return;
    }

    // Ensure a working/output folder is configured and exists. If missing,
    // prompt the user once and persist the choice to data/job_config.json.
    try {
        var cfgPath = projectRoot.fsName + "/data/job_config.json";
        var cfgFile = new File(cfgPath);
        var cfg = {};
        if (cfgFile.exists) {
            try {
                cfgFile.open('r');
                var txt = cfgFile.read();
                cfgFile.close();
                cfg = JSON.parse(txt || '{}');
            } catch (e) {
                try { cfgFile.close(); } catch (e2) {}
                cfg = {};
            }
        }

        var outVal = cfg.output_dir || "jobs";
        // Resolve possible locations: absolute or relative to projectRoot
        var outFolder = new Folder(outVal);
        if (!outFolder.exists) {
            var rel = new Folder(projectRoot.fsName + "/" + outVal);
            if (rel.exists) outFolder = rel;
        }

        if (!outFolder.exists) {
            // Prompt user once to pick a working/output folder
            var picked = Folder.selectDialog("Select working/output folder for job files (will be saved)");
            if (picked == null) {
                alert("Working folder not configured. Image mapping canceled.");
                return;
            }
            outFolder = picked;

            // persist into job_config.json (use absolute path)
            cfg.output_dir = outFolder.fsName;
            try {
                cfgFile.open('w');
                cfgFile.encoding = 'UTF-8';
                cfgFile.write(JSON.stringify(cfg, null, 2));
                cfgFile.close();
            } catch (e) {
                try { cfgFile.close(); } catch (e2) {}
            }
        }

        // Ensure folder exists on disk
        try { if (!outFolder.exists) outFolder.create(); } catch (e) {}

        $.writeln("Working folder resolved: " + outFolder.fsName);
    } catch (e) {
        $.writeln("Working folder resolution failed: " + e);
    }

    // Run the python controller hidden (avoid opening a CMD window).
    // Create a small VBScript that executes Python and echoes stdout.
    var vbsFile = new File(projectRoot.fsName + "/jobs/run_controller.vbs");
    var pythonPath = projectRoot.fsName + "/.venv/Scripts/python.exe";
    if (!new File(pythonPath).exists) pythonPath = "python"; // fallback

    var vbsCmd = [];
    // write output to a temp file, run hidden (windowStyle=0), then echo the file
    var outFile = projectRoot.fsName + "/jobs/run_output.txt";
    // Build a safe command line (quote paths) and redirect stdout/stderr to outFile
    var pyCmd = '"' + pythonPath + '" "' + pyController.fsName + '" "' + selectedVoiceover.fsName + '" > "' + outFile + '" 2>&1';
    vbsCmd.push('Set objShell = CreateObject("WScript.Shell")');
    vbsCmd.push('Set fso = CreateObject("Scripting.FileSystemObject")');
    vbsCmd.push('cmd = "' + pyCmd.replace(/"/g, '""') + '"');
    vbsCmd.push('objShell.Run cmd, 0, True');
    vbsCmd.push('s = ""');
    vbsCmd.push('If fso.FileExists("' + outFile.replace(/"/g, '""') + '") Then');
    vbsCmd.push('  Set tf = fso.OpenTextFile("' + outFile.replace(/"/g, '""') + '", 1)');
    vbsCmd.push('  s = tf.ReadAll()');
    vbsCmd.push('  tf.Close');
    vbsCmd.push('  fso.DeleteFile("' + outFile.replace(/"/g, '""') + '")');
    vbsCmd.push('End If');
    vbsCmd.push('WScript.Echo s');

    try {
        vbsFile.open('w');
        vbsFile.encoding = 'UTF-8';
        vbsFile.write(vbsCmd.join('\r\n'));
        vbsFile.close();
    } catch (e) {
        alert('Failed to write temporary runner script: ' + e);
        return;
    }

    var result = system.callSystem('cscript //Nologo "' + vbsFile.fsName + '"');
    // cleanup
    try { vbsFile.remove(); } catch (e) {}
    if (!result) {
        alert("Failed to generate timeline. See console for details.");
        return;
    }

    // The controller may print diagnostic lines. First try to find the
    // explicit TIMELINE_PATH= marker. If not present, fall back to scanning
    // for an absolute Windows path that ends with timeline.json.
    var timelineFile = null;
    try {
        var m = result.match(/TIMELINE_PATH=(.*)/);
        if (m && m[1]) {
            var cand = m[1].trim().replace(/^\"|\"$/g, "");
            var f = new File(cand);
            if (f.exists) timelineFile = f;
        }
        if (!timelineFile) {
            // fallback: look for C:\...\timeline.json or similar
            var re = /[A-Za-z]:\\[^\n\r]*?timeline\.json/gi;
            var found = result.match(re) || [];
            for (var i = 0; i < found.length; i++) {
                try {
                    var c = found[i].trim();
                    var ff = new File(c);
                    if (ff.exists) { timelineFile = ff; break; }
                } catch (e) {}
            }
        }
    } catch (e) {
        timelineFile = null;
    }

    if (!timelineFile) {
        // final fallback: known default path under project `jobs` folder
        try {
            var defPath = projectRoot.fsName + "/jobs/timeline.json";
            var defFile = new File(defPath);
            if (defFile.exists) {
                timelineFile = defFile;
            }
        } catch (e) {}

        if (!timelineFile) {
            alert("Timeline JSON not found:\n" + result);
            return;
        }
    }

    $.writeln("Image mapping start");
    timelineFile.open("r");
    var data = JSON.parse(timelineFile.read());
    timelineFile.close();

    // Import audio first and set comp duration to audio duration
    var audioFile = new File(data.audio.src);
    var audioItem = null;
    var compDuration = 300;
    if (audioFile.exists) {
        audioItem = app.project.importFile(new ImportOptions(audioFile));
        compDuration = audioItem.duration || compDuration;
    }

    // set comp size from meta if provided, otherwise default
    var compWidth = (data.meta && data.meta.width) ? data.meta.width : 1080;
    var compHeight = (data.meta && data.meta.height) ? data.meta.height : 1920;

    var comp = app.project.items.addComp(
        "AUTO_COMP",
        compWidth,
        compHeight,
        1,
        compDuration,
        data.meta.fps
    );
    // Set preview quality to quarter
    try {
        comp.resolution = CompItem.RESOLUTION_QUARTER;
    } catch (e) {
        // fallback for older AE versions
        comp.resolution = 2;
    }

    if (audioItem) {
        comp.layers.add(audioItem);
    }

    // Ensure cache directory exists
    var cachePath = (data.cache_dir) ? data.cache_dir : (projectRoot.fsName + "/jobs/cache");
    var cacheFolder = new Folder(cachePath);
    if (!cacheFolder.exists) cacheFolder.create();

    // Import images once and create one layer per spoken word (precise timing).
    var footageCache = {};
    function importSrcToFootage(src, idx) {
        if (footageCache[src]) return footageCache[src];
        var fileToImport = null;

        if (/^https?:\/\//i.test(src)) {
            var parts = src.split('/');
            var last = parts[parts.length - 1].split('?')[0];
            var safeName = idx + "_" + last;
            var localPath = cacheFolder.fsName + "/" + safeName;
            var localFile = new File(localPath);

            if (!localFile.exists) {
                var psCmd = 'powershell -NoProfile -Command "Try { (New-Object System.Net.WebClient).DownloadFile(\\"' + src + '\\", \\\"' + localFile.fsName + '\\\") } Catch { exit 1 }"';
                try { system.callSystem(psCmd); } catch (e) {}
            }

            if (localFile.exists) fileToImport = localFile;
        } else {
            var possibleFile = new File(src);
            if (possibleFile.exists) fileToImport = possibleFile;
        }

        if (!fileToImport) {
            footageCache[src] = null;
            return null;
        }

        try {
            var footage = app.project.importFile(new ImportOptions(fileToImport));
            footageCache[src] = footage;
            return footage;
        } catch (e) {
            footageCache[src] = null;
            return null;
        }
    }

    // Use `data.words` and `segment.word_indices` to place layers exactly at word boundaries.
    var words = data.words || [];
    for (var si = 0; si < data.segments.length; si++) {
        var seg = data.segments[si];
        var src = seg.src;
        var isArray = Object.prototype.toString.call(src) === '[object Array]';

        var footageA = null, footageB = null, footageSingle = null;
        if (isArray) {
            footageA = importSrcToFootage(src[0], si + '_a');
            footageB = importSrcToFootage(src[1], si + '_b');
            if (!footageA && !footageB) continue;
        } else {
            footageSingle = importSrcToFootage(src, si);
            if (!footageSingle) continue;
        }

        var wordIdxs = seg.word_indices || [];
        if (!wordIdxs.length) {
            // fallback: create single layer for whole segment
            if (isArray) {
                // create two side-by-side layers
                if (footageA) {
                    var lA = comp.layers.add(footageA);
                    lA.startTime = seg.start; lA.outPoint = seg.end;
                    lA.property('Position').setValueAtTime(seg.start, [compWidth * 0.25, compHeight * 0.5]);
                    lA.property('Position').setValueAtTime(seg.end, [compWidth * 0.25, compHeight * 0.5]);
                }
                if (footageB) {
                    var lB = comp.layers.add(footageB);
                    lB.startTime = seg.start; lB.outPoint = seg.end;
                    lB.property('Position').setValueAtTime(seg.start, [compWidth * 0.75, compHeight * 0.5]);
                    lB.property('Position').setValueAtTime(seg.end, [compWidth * 0.75, compHeight * 0.5]);
                }
            } else {
                var l = comp.layers.add(footageSingle);
                l.startTime = seg.start;
                l.outPoint = seg.end;
                if (seg.animation === "zoom_in") {
                    l.property("Scale").setValueAtTime(seg.start, [100,100]);
                    l.property("Scale").setValueAtTime(seg.end, [110,110]);
                }
                if (seg.animation === "slide_left") {
                    l.property("Position").setValueAtTime(seg.start, [compWidth * 0.55, compHeight * 0.5]);
                    l.property("Position").setValueAtTime(seg.end, [compWidth * 0.45, compHeight * 0.5]);
                }
            }
            continue;
        }

        for (var wi = 0; wi < wordIdxs.length; wi++) {
            var widx = wordIdxs[wi];
            var gw = words[widx];
            if (!gw) continue;
            if (isArray) {
                if (footageA) {
                    var la = comp.layers.add(footageA);
                    la.startTime = gw.start; la.outPoint = gw.end;
                    la.property('Position').setValueAtTime(gw.start, [compWidth * 0.25, compHeight * 0.5]);
                    la.property('Position').setValueAtTime(gw.end, [compWidth * 0.25, compHeight * 0.5]);
                }
                if (footageB) {
                    var lb = comp.layers.add(footageB);
                    lb.startTime = gw.start; lb.outPoint = gw.end;
                    lb.property('Position').setValueAtTime(gw.start, [compWidth * 0.75, compHeight * 0.5]);
                    lb.property('Position').setValueAtTime(gw.end, [compWidth * 0.75, compHeight * 0.5]);
                }
            } else {
                var lw = comp.layers.add(footageSingle);
                lw.startTime = gw.start;
                lw.outPoint = gw.end;

                // apply same animation but per-word
                if (seg.animation === "zoom_in") {
                    lw.property("Scale").setValueAtTime(gw.start, [100,100]);
                    lw.property("Scale").setValueAtTime(gw.end, [110,110]);
                }
                if (seg.animation === "slide_left") {
                    lw.property("Position").setValueAtTime(gw.start, [compWidth * 0.55, compHeight * 0.5]);
                    lw.property("Position").setValueAtTime(gw.end, [compWidth * 0.45, compHeight * 0.5]);
                }
            }
        }
    }

    $.writeln("Image mapping complete");
    alert("Timeline built. Preview and render.");
    app.endUndoGroup();
})();
