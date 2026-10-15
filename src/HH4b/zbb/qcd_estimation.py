#!/usr/bin/env python3
# QCD estimation of the scouting Zbb pass regions, adapted from the VHcc had channel one (Beatriz-PlotTools).
# Inputs are the templates_{year}_{pT bin}.root of scouting_templates.py, with histograms {sample}_{region}.
# Differences to the VHcc one: these templates have no low V mass control region, so the TF is fitted to the pass/fail
# ratio of the same pT bin, with the H signal region blinded (the pass data of scouting_templates.py is zeroed there,
# BLIND_WINDOW). The fail * TF estimate is normalised with one factor, so that it equals the pass outside the blinded
# window: the VHcc normalisation (to the fail integral, then times epsilon) assumes a TF flat across the window.
import ROOT
import numpy as np
import argparse
import glob
import os
import re

ROOT.gROOT.SetBatch(True)

# -------------------------
# helper functions
# -------------------------
def subtract_non_qcd(hist, file, region):
    """(data - nonQCD MC)"""
    data_minus_MC = hist.Clone()
    data_minus_MC.SetDirectory(0)
    mc_names = ["Zto2Q_BB", "Zto2Q_CC", "Zto2Q_QQ", "Wto2Q", "ttbar", "singletop"]  # bkg_keys of scouting_templates.py
    for mc in mc_names:
        h = file.Get(f"{mc}_{region}")
        if h:
            data_minus_MC.Add(h, -1)
    return data_minus_MC

def blind_sigRegion(hist, low=100, high=150):
    """blind Hbb region by default (100-150), as BLIND_WINDOW of scouting_templates.py"""
    newhist = hist.Clone(hist.GetName()+"_blind")
    for i in range(1, newhist.GetNbinsX() + 1):
        x_axis = newhist.GetBinCenter(i)
        if low < x_axis < high:
            newhist.SetBinContent(i, 0)
            newhist.SetBinError(i, 0)
    return newhist

# -------------------------
# CLI
# -------------------------
parser = argparse.ArgumentParser('qcd estimation (Zbb)')
parser.add_argument('--basedir', help='Input directory (output directory of scouting_templates.py)',
                    default="/eos/user/z/zima/scouting/templates/nanotrees_tagger_Cali_2024_had_vh")
parser.add_argument('--year', default="2024All")
parser.add_argument('--fail-region', default=None,
                    help='fail region for the QCD shape. Default: the fail_TXbb* region in the templates, i.e. '
                         'QCD_FAIL_REGION of scouting_templates.py (TXbb from 0.4 to the fail upper bound)')
options = parser.parse_args()
basedir = options.basedir
year = options.year
fail_region = options.fail_region

# -------------------------
# category configuration: every pass region of every pT bin
# templates file : templates_{year}_{pT bin}.root
# pass           : {sample}_pass_TXbb{low}to{high}_{pT bin}
# fail (QCD)     : {sample}_{fail_region}, TXbb 0.4 to the fail upper bound (the fit's "fail" starts at 0)
# -------------------------
categories = {}
template_files = sorted(glob.glob(os.path.join(basedir, f"templates_{year}_pT*.root")))
if not template_files:
    raise FileNotFoundError(f"No templates_{year}_pT*.root in {basedir}")
for fname in template_files:
    f = ROOT.TFile.Open(fname)
    for key in f.GetListOfKeys():
        m = re.fullmatch(r"data_(pass_TXbb.*_pT\d+to\d+)", key.GetName().split(";")[0])
        if m:
            categories[m.group(1)] = {"file": fname}
        # the QCD fail region, whose name follows the TXbb bins of the templates
        #m = re.fullmatch(r"data_(fail_TXbb[^_]+to[^_]+)", key.GetName().split(";")[0])
        m = re.fullmatch(r"data_(fail)", key.GetName().split(";")[0])

        if m and fail_region is None:
            fail_region = m.group(1)
    f.Close()

if fail_region is None:
    raise RuntimeError(f"No fail_TXbb* region in the templates of {basedir}, give it with --fail-region")
print(f"QCD fail region: {fail_region}")

flavor = "Zbb"

# -------------------------
# main loop over categories
# -------------------------
for cat_name, cat_cfg in categories.items():
    print(f"\n=== Processing {flavor} category: {cat_name} ===")

    # pass and fail histograms are in the same templates file
    file_TEMPLATES = ROOT.TFile.Open(cat_cfg["file"])

    if not file_TEMPLATES:
        print("Error opening ROOT file for", cat_name, flavor)
        continue

    # (data - nonQCD MC) pass/fail -> TF and epsilon
    H_qcd_pass_hist = subtract_non_qcd(file_TEMPLATES.Get(f"data_{cat_name}"), file_TEMPLATES, cat_name)
    H_qcd_fail_hist = subtract_non_qcd(file_TEMPLATES.Get(f"data_{fail_region}"), file_TEMPLATES, fail_region)

    # binning (assume consistent)
    xmin = H_qcd_fail_hist.GetXaxis().GetXmin()
    xmax = H_qcd_fail_hist.GetXaxis().GetXmax()
    nbins = H_qcd_fail_hist.GetNbinsX()

    # blind Hbb signal region (100-150) for the TF fit and epsilon calculation
    H_goodV_pass_blind = blind_sigRegion(H_qcd_pass_hist)
    H_qcd_fail_blind = blind_sigRegion(H_qcd_fail_hist)

    # pass/fail ratio from the blinded histograms (blinded bins are empty and skipped by the fit)
    ratio = H_goodV_pass_blind.Clone(f"ratio_{flavor}_{cat_name}")
    ratio.SetDirectory(0)
    ratio.Divide(H_qcd_fail_blind)

    # quadratic TF
    x_min = ratio.GetXaxis().GetXmin()
    x_max = ratio.GetXaxis().GetXmax()
    fit_func = ROOT.TF1(f"fit_{flavor}_{cat_name}", "[0] + [1]*x + [2]*x**2", x_min, x_max)

    # epsilon from the blinded histograms
    H_N_pass = H_goodV_pass_blind.Integral()
    H_N_fail = H_qcd_fail_blind.Integral()
    eps = H_N_pass / H_N_fail if H_N_fail > 0 else 0.0
    print(f"{flavor} epsilon (category {cat_name}) = {eps:.4f}")

    # fit ratio, chi2 with the ratio errors ("W" would ignore them: equal weights and meaningless parameter errors)
    f = ratio.Fit(fit_func, "S")
    p0, p1, p2 = f.Parameter(0), f.Parameter(1), f.Parameter(2)
    p0_err, p1_err, p2_err = f.ParError(0), f.ParError(1), f.ParError(2)
    cov = f.GetCovarianceMatrix()
    cov_p0_p1 = cov[0][1]
    cov_p0_p2 = cov[0][2]
    cov_p1_p2 = cov[1][2]

    # apply TF bin-by-bin to the fail histogram
    nbins_ratio = ratio.GetNbinsX()
    x_centers = np.array([ratio.GetBinCenter(i) for i in range(1, nbins_ratio + 1)])

    contents = [H_qcd_fail_hist.GetBinContent(i) for i in range(1, nbins + 1)]
    errors   = [H_qcd_fail_hist.GetBinError(i)   for i in range(1, nbins + 1)]

    qcd_TF_list = []
    qcd_TF_errs = []
    for cnt, err, x in zip(contents, errors, x_centers):
        TF = p0 + p1 * x + p2 * x * x
        qcd_val = cnt * TF
        sigma_TF2 = (
            p0_err**2 +
            (x**2) * p1_err**2 +
            (x**4) * p2_err**2 +
            2 * x * cov_p0_p1 +
            2 * (x**2) * cov_p0_p2 +
            2 * (x**3) * cov_p1_p2
        )
        sigma2 = (TF**2) * (err**2) + (cnt**2) * sigma_TF2
        qcd_err = np.sqrt(sigma2) if sigma2 > 0 else 0.0
        qcd_TF_list.append(qcd_val)
        qcd_TF_errs.append(qcd_err)

    # qcd TF hist
    qcd_TF_hist = ROOT.TH1F(f"qcd_TF_{flavor}_{cat_name}", f"QCD {flavor} fail * TF {cat_name}", nbins, xmin, xmax)
    qcd_TF_hist.SetDirectory(0)
    for i, (val, er) in enumerate(zip(qcd_TF_list, qcd_TF_errs), start=1):
        qcd_TF_hist.SetBinContent(i, val)
        qcd_TF_hist.SetBinError(i, er)

    # normalize fail * TF to the pass outside the blinded window: sum_out(pass) / sum_out(fail * TF)
    int_pass_out = H_goodV_pass_blind.Integral()
    int_qcdtf_out = blind_sigRegion(qcd_TF_hist).Integral()
    norm = int_pass_out / int_qcdtf_out if int_qcdtf_out > 0 else 0.0
    qcd_TF_hist.Scale(norm)
    print(f"{flavor} normalisation (category {cat_name}) = {norm:.4f}")

    # fail scaled by epsilon, for the plot
    H_qcd_fail_scaled = H_qcd_fail_hist.Clone(f"{flavor}_{cat_name}_fail_scaled_eps")
    H_qcd_fail_scaled.SetDirectory(0)
    H_qcd_fail_scaled.Scale(eps)

    # plot the pass data blinded: it is zeroed in the templates, so data - MC is negative there
    H_qcd_pass_hist = H_goodV_pass_blind

    # canvas and legend
    can = ROOT.TCanvas(f"can_{flavor}_{cat_name}", "can", 1000, 750)
    legend = ROOT.TLegend(0.45, 0.65, 0.85, 0.85)
    legend.SetBorderSize(0)
    legend.SetFillStyle(0)
    legend.SetTextSize(0.035)

    H_qcd_pass_hist.SetMarkerStyle(20)
    H_qcd_pass_hist.SetMarkerSize(1)
    H_qcd_pass_hist.SetMarkerColor(ROOT.kRed+1)
    H_qcd_pass_hist.SetLineColor(ROOT.kRed+1)
    H_qcd_pass_hist.SetLineWidth(1)
    H_qcd_pass_hist.GetYaxis().SetRangeUser(0, max(30, H_qcd_pass_hist.GetMaximum()*1.2))

    H_qcd_fail_scaled.SetMarkerStyle(21)
    H_qcd_fail_scaled.SetMarkerSize(1)
    H_qcd_fail_scaled.SetMarkerColor(ROOT.kBlue+1)
    H_qcd_fail_scaled.SetLineColor(ROOT.kBlue+1)
    H_qcd_fail_scaled.SetLineWidth(1)

    qcd_TF_hist.SetLineColor(ROOT.kGreen-6)
    qcd_TF_hist.SetFillColorAlpha(ROOT.kGreen+1, 0.1)
    qcd_TF_hist.SetLineWidth(2)

    H_qcd_pass_hist.Draw("E1")
    H_qcd_fail_scaled.Draw("E1 SAME")
    qcd_TF_hist.Draw("HIST SAME")
    qcd_TF_hist.Draw("E1 SAME")

    legend.AddEntry(H_qcd_pass_hist, "data pass", "pe")
    legend.AddEntry(H_qcd_fail_scaled, "data fail #times epsilon", "pe")
    legend.AddEntry(qcd_TF_hist, "data fail #times TF (norm. to pass)", "lf")
    legend.Draw()

    H_qcd_pass_hist.SetTitle(f"QCD estimation {flavor} {cat_name}")
    H_qcd_pass_hist.GetXaxis().SetTitle("m^{scoutGloParT}_{res} (GeV)")
    H_qcd_pass_hist.GetYaxis().SetTitle(f"Events/{H_qcd_pass_hist.GetBinWidth(1):.2f} GeV")
    ROOT.gStyle.SetOptStat(0)
    can.Update()

    # outputs
    pdf_name = f"{flavor}_{cat_name}_QCDestimation.pdf"
    txt_name = f"{flavor}_{cat_name}_QCDestimation.txt"
    root_out_name = cat_cfg["file"]
    try:
        can.SaveAs(pdf_name)
    except Exception as e:
        print("Warning saving pdf:", e)

    qcd_content = [qcd_TF_hist.GetBinContent(i) for i in range(1, nbins + 1)]
    qcd_errors  = [qcd_TF_hist.GetBinError(i)   for i in range(1, nbins + 1)]
    with open(txt_name, "w") as ftxt:
        ftxt.write("=== QCD Estimation Results ===\n")
        ftxt.write(f"Category: {cat_name}\n")
        ftxt.write(f"Flavor: {flavor}\n")
        ftxt.write(f"Epsilon: {eps:.6f}\n")
        ftxt.write(f"Normalisation of fail * TF (pass / fail * TF outside the blinded window): {norm:.6f}\n")
        ftxt.write(f"TF params: p0={p0:.6e} +/- {p0_err:.6e}, p1={p1:.6e} +/- {p1_err:.6e}, p2={p2:.6e} +/- {p2_err:.6e}\n")
        ftxt.write(f"QCD estimated (bins): {qcd_content}\n")
        ftxt.write(f"Errors in estimated QCD (bins): {qcd_errors}\n")

    file_TEMPLATES.Close()

    # write qcd_estimated into the templates file, named like the other templates: qcd_{region}
    fout = ROOT.TFile.Open(root_out_name, "UPDATE")
    if fout and not fout.IsZombie():
        qcd_TF_hist.SetName(f"qcd_{cat_name}")
        qcd_TF_hist.Write("", ROOT.TObject.kOverwrite)
        fout.Close()
        print(f"Wrote qcd_{cat_name} to", root_out_name)
    else:
        print("Warning: couldn't open output root file", root_out_name)

print(f"\nAll {flavor} categories processed.")
