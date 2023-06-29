



def test_spst():
    from sps.spst import test_SPP_PSS
    test_SPP_PSS()

def test_osps_dp():
    from sps.osps_dp import test_OSPS_DP
    f = test_OSPS_DP()
    print(f)

if __name__ == '__main__':
    # test_spst()
    test_osps_dp()
