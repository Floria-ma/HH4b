#!/usr/bin/env python3
# QCD estimation of the scouting Zbb pass regions, adapted from the VHcc had channel one (Beatriz-PlotTools).
# Inputs are the templates_{year}_{pT bin}.root of scouting_templates.py, with histograms {sample}_{region}.
# Differences to the VHcc one: these templates have no low V mass control region, so the TF is fitted to the pass/fail
# ratio of the same pT bin, with the H signal region blinded (the pass data of scouting_templates.py is zeroed there,
# BLIND_WINDOW). The fail * TF estimate is normalised with one factor, so that it equals the pass outside the blinded
# window: the VHcc normalisation (to the fail integral, then times epsilon) assumes a TF flat across the window.
import ROOT
import numpy as np
import uproot
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
    mc_names = ["Zto2Q_BB", "Zto2Q_CC", "Zto2Q_QQ", "Wto2Q", "ttbar", "singletop", "hbb"]  # bkg_keys and sig_keys of scouting_templates.py
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
parser.add_argument('--fail-region', default="fail",
                    help='fail region for the QCD shape. Default: the full fail region of scouting_templates.py '
                         '(TXbb from 0 to the fail upper bound)')
parser.add_argument('--tf', choices=["quadratic", "linear"], default="quadratic",
                    help='polynomial in the mass of the TF fitted to the pass/fail ratio')
options = parser.parse_args()
basedir = options.basedir
year = options.year
fail_region = options.fail_region

# -------------------------
# category configuration: every pass region of every pT bin
# templates file : templates_{year}_{pT bin}.root
# pass           : {sample}_pass_TXbb{low}to{high}_{pT bin}
# fail (QCD)     : {sample}_{fail_region}, TXbb 0 to the fail upper bound
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
    f.Close()

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

    # linear or quadratic TF
    x_min = ratio.GetXaxis().GetXmin()
    x_max = ratio.GetXaxis().GetXmax()
    tf_formula = "[0] + [1]*x" if options.tf == "linear" else "[0] + [1]*x + [2]*x**2"
    fit_func = ROOT.TF1(f"fit_{flavor}_{cat_name}", tf_formula, x_min, x_max)
    n_par = fit_func.GetNpar()

    # epsilon from the blinded histograms
    H_N_pass = H_goodV_pass_blind.Integral()
    H_N_fail = H_qcd_fail_blind.Integral()
    eps = H_N_pass / H_N_fail if H_N_fail > 0 else 0.0
    print(f"{flavor} epsilon (category {cat_name}) = {eps:.4f}")

    # fit ratio, chi2 with the ratio errors ("W" would ignore them: equal weights and meaningless parameter errors)
    f = ratio.Fit(fit_func, "S")
    pars = np.array([f.Parameter(i) for i in range(n_par)])
    par_errs = np.array([f.ParError(i) for i in range(n_par)])
    cov_matrix = f.GetCovarianceMatrix()
    cov = np.array([[cov_matrix[i][j] for j in range(n_par)] for i in range(n_par)])

    # apply TF bin-by-bin to the fail histogram
    nbins_ratio = ratio.GetNbinsX()
    x_centers = np.array([ratio.GetBinCenter(i) for i in range(1, nbins_ratio + 1)])

    contents = [H_qcd_fail_hist.GetBinContent(i) for i in range(1, nbins + 1)]
    errors   = [H_qcd_fail_hist.GetBinError(i)   for i in range(1, nbins + 1)]

    qcd_TF_list = []
    qcd_TF_errs = []
    for cnt, err, x in zip(contents, errors, x_centers):
        # TF = sum_k p_k x^k, sigma_TF^2 = v^T cov v with v = (1, x, x^2, ...)
        powers = x ** np.arange(n_par)
        TF = pars @ powers
        qcd_val = cnt * TF
        sigma_TF2 = powers @ cov @ powers
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
        ftxt.write(f"TF: {options.tf}\n")
        par_strs = [f"p{i}={p:.6e} +/- {e:.6e}" for i, (p, e) in enumerate(zip(pars, par_errs))]
        ftxt.write(f"TF params: {', '.join(par_strs)}\n")
        ftxt.write(f"QCD estimated (bins): {qcd_content}\n")
        ftxt.write(f"Errors in estimated QCD (bins): {qcd_errors}\n")

    file_TEMPLATES.Close()

    # write qcd_estimated into the templates file, named like the other templates: qcd_{region}.
    # With uproot, as the file was written by uproot: a ROOT "UPDATE" of it overwrites other objects in the file
    # (uproot.from_pyroot fails with this ROOT, so the TH1 is rebuilt from its bins, flow bins and errors included)
    n = qcd_TF_hist.GetNbinsX()
    stats = np.zeros(4)
    qcd_TF_hist.GetStats(stats)
    qcd_th1 = uproot.writing.identify.to_TH1x(
        fName=f"qcd_{cat_name}",
        fTitle=qcd_TF_hist.GetTitle(),
        data=np.array([qcd_TF_hist.GetBinContent(i) for i in range(n + 2)], dtype=np.float64),
        fEntries=qcd_TF_hist.GetEntries(),
        fTsumw=stats[0],
        fTsumw2=stats[1],
        fTsumwx=stats[2],
        fTsumwx2=stats[3],
        fSumw2=np.array([qcd_TF_hist.GetBinError(i) ** 2 for i in range(n + 2)], dtype=np.float64),
        fXaxis=uproot.writing.identify.to_TAxis(
            fName="xaxis", fTitle="", fNbins=n, fXmin=qcd_TF_hist.GetXaxis().GetXmin(), fXmax=qcd_TF_hist.GetXaxis().GetXmax()
        ),
    )
    # The file is rewritten instead of uproot.update: deleting and rewriting a key with uproot.update can leave a
    # duplicated end segment in the free list, after which uproot.update fails on the file (AssertionError)
    with uproot.open(root_out_name) as fin:
        hists = {k: fin[k] for k in fin.keys(cycle=False) if k != f"qcd_{cat_name}"}
    hists[f"qcd_{cat_name}"] = qcd_th1
    with uproot.recreate(root_out_name) as fout:
        for k, h in hists.items():
            fout[k] = h
    print(f"Wrote qcd_{cat_name} to", root_out_name)

print(f"\nAll {flavor} categories processed.")
