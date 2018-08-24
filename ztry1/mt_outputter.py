# to output the results
from zl.search import State, SearchGraph
from zl import data, utils
import numpy as np

class Outputter(object):
    def __init__(self, opts):
        self.opts = opts
        self.unk_rep = opts["decode_replace_unk"]
        self.inst_count = 0
        self.sent_count = 0
        self.replaced = 0

    @staticmethod
    def z2h(w):
        # todo(warn): especially for JP->EN
        r = ""
        for c in w:
            nn = ord(c)
            n2 = nn
            if nn == 12288:
                n2 = 32
            elif nn >= 65281 and nn <= 65374:
                n2 = nn - 65248
            r += chr(n2)
        return w

    def report(self):
        utils.zlog("Outputter final: count=%s/%s, replaced=%s" % (self.inst_count, self.sent_count, self.replaced))

    def transform_src(self, w):
        w = Outputter.z2h(w)
        if all(str.isalnum(z) or z=="." for z in w):
            return w
        else:
            return "<unkown>"

    def format(self, states, target_dict, kbest, verbose, reverse=False):
        ret = ""
        if not kbest:
            states = [states[0]]
        # if verbose:
        #     ret += "# znumber%s" % self.inst_count + "\n"
            # ret += states[0].sg.show_graph(target_dict, False) + "\n"
        self.inst_count += 1
        for s in states:
            self.sent_count += 1
            # list of states including eos
            paths = s.get_path()
            utils.zcheck(paths[-1].action_code==target_dict.eos, "Not ended with EOS!")
            out_list = paths[:-1]
            if kbest and len(out_list)==0:
                # todo(warn): avoid empty line for kbest outputs
                out_list = paths
            if reverse:
                out_list = reversed(out_list)
            for one in out_list:
                tmp_code = one.action_code
                unk_replace = False
                if not self.unk_rep or tmp_code != target_dict.unk:
                    ret += target_dict.getw(one.action_code)
                else:
                    # todo: replacing unk with bilingual dictionary
                    unk_replace = True
                    xwords = one.get("attention_src")
                    xidx = np.argmax(one.get("attention_weights"))
                    if xidx >= len(xwords):
                        # utils.zcheck(False, "attention out of range", func="warn")
                        rrw = "<out-of-range>"
                        # rrw = ""
                    else:
                        rrw = xwords[xidx]
                        self.replaced += 1
                    ret += self.transform_src(rrw)
                if verbose:
                    if unk_replace:
                        ret += "<UNK>"
                    ret += ("|%.3f" % one.action_score())
                ret += " "
            # last one
            if verbose:
                one = paths[-1]
                ret += target_dict.getw(one.action_code)
                ret += ("|%.3f" % one.action_score())
                ret += "FINALLY|%d|%.3f|%.3f" % (one.length, one.score_partial, one.score_final)
            ret += "\n"
        if kbest:
            ret += "\n"
        return ret
