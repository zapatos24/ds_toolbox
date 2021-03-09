import requests
import time
import sys


class API_Handler:
    def __init__(self):
        """
        Initializes the api object, setting the base url to v1/projects and defines the append for a search query. It
        also initializes an empty result_list dictionary to be filled in the construct_list function.
        """

        self.base_url = 'https://api.federalreporter.nih.gov/v1/projects/'
        self.search_start = 'search?query='
        self.result_list = []

    @staticmethod
    def get_keys(file_path):
        """
        Pulls necessary api keys from designated file path
        """
        with open(file_path) as f:
            return json.load(f)

    @staticmethod
    def url_construct(base_url, search_start, param_dict, offset=1, limit=50):
        """
        Takes in the base url for the api, the starting string to perform a search, as well as the
        parameters to pass to the api. The page offset and limit are optional arguments, set to 1 and 50 respectively
        if not explicitly stated in the call.
        :param base_url: The base url for the api
        :param search_start: The string to start a search (for the NIH api, it's just 'search?query=')
        :param param_dict: The dictionary of parameters to be passed in the api call. All keys must be strings, and all
        values must be lists
        :param offset: The page offset to send in the api call
        :param limit: The cap on the number of responses from the api call
        :return: the complete url string with base, search language, parsed parameters, offset, and limit joined
        """

        url = base_url + search_start
        for param in param_dict.keys():
            full_param_string = param + ':' + ','.join(param_dict[param]) + '$'
            url += full_param_string
        offset_string = '&offset=' + str(offset)
        limit_string = '&limit=' + str(limit)
        url += offset_string + limit_string
        return url

    @staticmethod
    def get_pages(url):
        """
        Takes the complete url for an api call and returns the number of pages that exist for that particular query
        :param url: The complete url, base, search, parameters, offset, and limit included.
        :return: The value for the 'totalPages' key from the api call
        """

        r = requests.get(url)
        return r.json()['totalPages']

    @staticmethod
    def progress(count, total, status=''):
        """
        A small progress bar to keep track of function status throughout the list building process.
        :param count: the count that is currently being run of the total
        :param total: the total count that we will reach
        :param status: a given status message to display next to the progress bar
        :return:
        """
        bar_len = 60
        filled_len = int(round(bar_len * count / float(total)))

        percents = round(100.0 * count / float(total), 1)
        bar = '=' * filled_len + '-' * (bar_len - filled_len)

        sys.stdout.write('[%s] %s%s ...%s\r' % (bar, percents, '%', status))
        sys.stdout.flush()

    def construct_list(self, param_dict):
        """
        Takes in the parameter dictionary and:
        - Calls the url_construct function to form a full url for use in an api call
        - Calls the get_pages function to determine how many total pages exist in this api call
        - Iteratively makes calls to the api, increasing the offset by the limit defined for each available page.
        Within the iterative process, a dictionary is created for each item returned from the api that extracts the
        pertinent information, and that dictionary is added to the result_list variable of the class. If an error occurs
        in the dictionary creation or appending operations, an error message is printed with the project number that
        failed.
        :param param_dict: The dictionary of the parameters to be sent via the api call. All keys should be strings, and
        all values should be lists.
        :return:
        """

        call = self.url_construct(self.base_url, self.search_start, param_dict)
        pages = self.get_pages(call)
        offset = 1
        limit = 50
        curr_page = 1
        # sys.stdout.write('Building award list...')
        while curr_page <= pages:
            self.progress(curr_page, pages, status='Building award list')
            call = self.url_construct(self.base_url, self.search_start, param_dict, offset=offset, limit=limit)
            r = requests.get(call)
            awards = r.json()['items']
            for award in awards:
                try:
                    df_row = {'project_num': award['projectNumber'],
                              'agency': award['agency'],
                              'title': award['title'],
                              'department': award['department'],
                              'fy': award['fy'],
                              'total_cost': award['totalCostAmount'],
                              'abstract': award['abstract'],
                              'org_state': award['orgState'],
                              'cong_district': award['congressionalDistrict']
                              }

                    self.result_list.append(df_row)

                except:
                    print('Could not parse project ' + award['projectNumber'])
            offset += limit
            curr_page += 1
            time.sleep(1)
