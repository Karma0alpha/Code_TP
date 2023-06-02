#!/usr/bin/python3
import sys
import subprocess
subprocess.call("mkdir 01.clean;mkdir 02.hisat;mkdir 03.kraken;mkdir 04.nr;mkdir 05.megahit;mkdir step2;mkdir plot;mkdir result;mkdir shell_hisat",shell=True)
file1 = open(sys.argv[1],"r")
file3 = open("kraken.sh","w")
file4 = open("braken.sh","w")
for line in file1:
    if line.strip("\n").split("\t")[-1] == "bf":
        refseq = "/mnt/noxa/db/bwa/Mnbat"
    elif line.strip("\n").split("\t")[-1] == "cw":
        refseq = "/mnt/noxa/db/bwa/ciwei/genome"
    elif line.strip("\n").split("\t")[-1] == "aw":
        refseq = "/mnt/noxa/db/bwa/aw"
    elif line.strip("\n").split("\t")[-1] == "bwyw":
        refseq = "/mnt/noxa/db/bwa/bwyw"
    elif line.strip("\n").split("\t")[-1] == "yj":
        refseq = "/mnt/noxa/db/bwa/chick/chick"
    elif line.strip("\n").split("\t")[-1] == "mh":
        refseq = "/mnt/noxa/db/bwa/genome"
    elif line.strip("\n").split("\t")[-1] == "qj":
        refseq = "/mnt/noxa/db/bwa/qunjing"
    elif line.strip("\n").split("\t")[-1] == "wz":
        refseq = "/mnt/noxa/db/bwa/wz"
    elif line.strip("\n").split("\t")[-1] == "bj":
        refseq = "/mnt/noxa/db/bwa/dove/dove"
    elif line.strip("\n").split("\t")[-1] == "yt":
        refseq = "/mnt/noxa/db/bwa/rabbit/genome"
    elif line.strip("\n").split("\t")[-1] == "pc":
        refseq = "/mnt/noxa/db/human/pc"
    elif line.strip("\n").split("\t")[-1] == "ss":
        refseq = "/mnt/noxa/db/human/shasu"
    elif line.strip("\n").split("\t")[-1] == "hjs":
        refseq = "/mnt/noxa/db/human/hjs"
    elif line.strip("\n").split("\t")[-1] == "kw":
        refseq = "/mnt/noxa/db/bwa/kw"
    elif line.strip("\n").split("\t")[-1] == "tz":
        refseq = "/mnt/noxa/db/bwa/tz"
    elif line.strip("\n").split("\t")[-1] == "yp":
        refseq = "/mnt/noxa/db/bwa/yp"
    elif line.strip("\n").split("\t")[-1] == "xp":
        refseq = "/mnt/noxa/db/bwa/xp"
    elif line.strip("\n").split("\t")[-1] == "lp":
        refseq = "/mnt/noxa/db/bwa/lp"
    elif line.strip("\n").split("\t")[-1] == "xtp":
        refseq = "/mnt/noxa/db/bwa/xtp"
    elif line.strip("\n").split("\t")[-1] == "zhaw":
        refseq = "/mnt/noxa/db/bwa/zhaw"
    elif line.strip("\n").split("\t")[-1] == "km":
        refseq = "/mnt/noxa/db/bwa/km"
    elif line.strip("\n").split("\t")[-1] == "hg" or line.strip("\n").split("\t")[-1] == "Y":
        refseq = "/mnt/noxa/db/human/genome"
    file2 = open("shell_hisat/B_"+line.split("\t")[0]+".sh","w")
    list1 = line.strip("\n").split("\t")
    step_fastp = "date >> B_{name}.log ; /mnt/software/bin/fastp -i {input1} -I {input2} -o {output1} -O {output2} -j {json} -h {html} -q 20 -u 30 -y -x -w 8 >> B_{name}.log;date >> B_{name}.log"
    step_fastp_y = "/mnt/software/bin/fastp -i {input1} -I {input2} -o {output1} -O {output2} -w 16"
    name = list1[0]
    input1 = list1[1]
    input2 = list1[2]
    type_sample = list1[3]
    output1 = "../01.clean/"+list1[0]+".clean.R1.fq.gz"
    output2 = "../01.clean/"+list1[0]+".clean.R2.fq.gz"
    json = "../01.clean/"+list1[0]+".json"
    html = "../01.clean/"+list1[0]+".html"
    if type_sample != "Y":
        file2.write(step_fastp.format_map(vars())+"\n")
        file2.write("python /mnt/software/json_py/quality_base_data.py ../01.clean/"+list1[0]+".json ../plot/"+list1[0]+".txt\n")
        file2.write("Rscript /mnt/software/Rscript/GC_percent.R ../plot/"+list1[0]+"\n")
        file2.write("Rscript /mnt/software/Rscript/quelity.R ../plot/"+list1[0]+"\n")
    else:
        file2.write(step_fastp_y.format_map(vars())+"\n")
    step_hisat = "date >> B_{name}.log;/mnt/software/bin/hisat2 -x "+refseq+" -1 {output1} -2 {output2} --un-gz {unparid_unmap} --un-conc-gz {parid_unmap} -S {sam} --summary-file {summary} --threads 16 >> B_{name}.log;date >> B_{name}.log"
    step_hisat_y = "/mnt/software/bin/hisat2 -x /mnt/noxa/db/human/hjs -1 {output1} -2 {output2} --un-gz {unparid_unmap} --un-conc-gz {parid_unmap} -S {sam} --threads 16"
    unparid_unmap = "../02.hisat/"+list1[0]+".unpaired.unmap.gz"
    parid_unmap = "../02.hisat/"+list1[0]+".pairs.unmap.gz"
    sam = "../02.hisat/"+list1[0]+".map.sam"
    summary = "../02.hisat/"+list1[0]+".summary.txt"
    if type_sample != "Y":
        file2.write(step_hisat.format_map(vars())+"\n")
    else:
        file2.write(step_hisat_y.format_map(vars())+"\n")
    step_kraken = "date >> time ; /mnt/software/kraken2/kraken2 --threads 32 --db /mnt/noxa/db/download/refseq/ --threads 60 --output {stat} --unclassified-out {unclass} --report {report} 02.hisat/{name}.pairs.unmap.1.gz --paired 02.hisat/{name}.pairs.unmap.2.gz;date >> time ;sleep 1m"
    stat = "03.kraken/"+name+".stat"
    unclass = "03.kraken/"+name+".unclassified#.fastq"
    report = "03.kraken/"+name+".report"
    file3.write(step_kraken.format_map(vars())+"\n")
    step_braken_g = "/mnt/software/Bracken/bracken -d /mnt/noxa/db/download/refseq/ -i {report} -o 03.kraken/{name}_g.braken -w 03.kraken/{name}_g.report -r 150 -l G"
    step_braken_s = "/mnt/software/Bracken/bracken -d /mnt/noxa/db/download/refseq/ -i {report} -o 03.kraken/{name}_s.braken -w 03.kraken/{name}_s.report -r 150 -l S"
#    step_sele = "python /mnt/software/script/select_report.py 03.kraken/"+name+"_s.report"+" result/"+name+".mpa"
#    step_sele_g = "python /mnt/software/script/select_genus_report.py 03.kraken/"+name+"_g.report"+" result/"+name+"_g.mpa"
    file4.write(step_braken_g.format_map(vars())+"\n")
    file4.write(step_braken_s.format_map(vars())+"\n")
#    file4.write(step_sele+"\n")
#    file4.write(step_sele_g+"\n")
file4.write("python3 /mnt/software/script/change_mpa.py sample kreport.sh\nbash kreport.sh\nls 02.hisat/*summary.txt > summary.list\npython3 /mnt/software/script/change_mpa.py sample kreport.sh\nbash kreport.sh\npython3 /mnt/software/script/combine.py sample combine.sh\nbash combine.sh\npython3 /mnt/software/script/se_class.py result/result.mpa result/result_s.mpa s\nsed -i 's#_s.report##g' result/result_s.mpa\npython3 /mnt/software/script/rm_yin.py result/result_s.mpa result/result_rm.mpa\npython3 /mnt/software/script/se_virus_report.py sample result/result_rm.mpa\npython3 /mnt/software/script/stat_qc_h.py sample clean.xls\npython3 /mnt/noxa/script/maphost_stat.py summary.list dup_data.xls\npython3 /mnt/noxa/script/class_data.py sample class_stat.xls\ncp /mnt/noxa/script/step2.py ./\npython step2.py sample $1")
