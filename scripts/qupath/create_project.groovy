import qupath.lib.projects.ProjectIO
import java.nio.file.Files
import java.nio.file.Paths

// ==== USER INPUTS (edit these paths) ====
def projectDir = new File("/Volumes/Finkbeiner-Steve/work/data/ALS/Qupath_annotated_projects/sals-lumbar")
def imagesDir  = new File("/Volumes/Finkbeiner-Steve/work/data/ALS/sals-lumbar")
def annotsDir  = new File("/Volumes/Finkbeiner-Steve/work/data/ALS/sals-lumbar/sals lumbar-latest-annotation")

// ==== 1. Create project ====
def project = createProject(projectDir, ProjectIO.ProjectType.DEFAULT)
println "Created project at: ${projectDir}"

// ==== 2. Add all .svs images ====
def svsFiles = imagesDir.listFiles({ f, name -> name.toLowerCase().endsWith(".svs") } as FilenameFilter)
if (svsFiles == null || svsFiles.length == 0) {
    println "⚠️ No SVS images found in ${imagesDir}"
} else {
    for (file in svsFiles) {
        project.addImage(file.toURI())
        println "Added image: ${file.getName()}"
    }
}

// ==== 3. Load .qpdata annotations (matched by basename) ====
def qpdataFiles = annotsDir.listFiles({ f, name -> name.toLowerCase().endsWith(".qpdata") } as FilenameFilter)
if (qpdataFiles == null || qpdataFiles.length == 0) {
    println "⚠️ No QPData files found in ${annotsDir}"
} else {
    for (qpdata in qpdataFiles) {
        def basename = qpdata.getName().replaceFirst(/\\.qpdata/, "")
        def entry = project.getImageList().find { it.getImageName().contains(basename) }
        if (entry != null) {
            def hierarchy = entry.readHierarchy()
            hierarchy.loadObjects(qpdata)
            entry.saveHierarchy(hierarchy)
            println "Loaded annotations for: ${basename}"
        } else {
            println "⚠️ No matching image for annotation file: ${qpdata.getName()}"
        }
    }
}

// ==== 4. Save project ====
project.syncChanges()
println "✅ Project ready: ${projectDir.getAbsolutePath()}"
